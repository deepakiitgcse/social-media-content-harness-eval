"""Custom harness v2: research-based rubric.

The model rates each Short 1-10 on categories drawn from published research
and YouTube's own statements (see docs/shorts_research.md). Scores are then
combined in code, not by the model:

  rubric-ranked    per category, z-score within the creator (how this Short
                   compares with the creator's other Shorts), weighted sum.
                   This is the main score: the label is creator-relative too.
  rubric-weighted  weighted average of raw category scores (no creator ranking).
  rubric-equal     creator-ranked, but equal weights.
"""

from __future__ import annotations

import statistics as st
from collections import defaultdict

from evalkit.models.gemini import Gemini, post_text, video_part, video_url
from evalkit.schema import Post

# (key, weight, definition, what a 1 looks like, what a 10 looks like)
CATEGORIES = [
    ("hook", 3.0,
     "How strongly the first 1-3 seconds stop a viewer from swiping: visual intrigue, a clear promise or question, works even on mute.",
     "slow or confusing start; nothing happens; easy to swipe past",
     "instantly arresting first frame and a clear reason to keep watching"),
    ("pacing", 2.0,
     "Retention through the middle: no dead air, steady new information or visual change, trimmed to essentials.",
     "long static stretches, repetition, padding",
     "every second earns its place; constant forward momentum"),
    ("payoff", 2.0,
     "Structure: sets up curiosity, tension or a question and delivers a satisfying payoff, punchline or reveal.",
     "no setup and no payoff; just footage",
     "clear setup with a strong, satisfying payoff"),
    ("emotion", 2.0,
     "Intensity of high-arousal emotion it evokes: amusement, awe, surprise, excitement, outrage.",
     "emotionally flat",
     "strong laugh, wow, or gasp reaction"),
    ("originality", 2.0,
     "Original creation with a fresh angle, vs. a re-upload or compilation of others' content or a generic template.",
     "re-uploaded or clipped from others, or an indistinguishable template",
     "clearly original content with a distinctive idea or voice"),
    ("value", 1.5,
     "Reason to like, comment or share: useful, relatable, or something you'd send to a friend.",
     "no takeaway and nothing to react to",
     "highly useful or relatable; begs to be shared or commented on"),
    ("audio", 1.0,
     "Audio actively drives the video: voice, music choice, sound design, timing to the beat.",
     "silent, noisy, or irrelevant background audio",
     "audio is integral and amplifies the content"),
    ("rewatch", 1.0,
     "Rewatchability: seamless loop, or dense or satisfying enough to replay.",
     "no reason to watch twice",
     "loops seamlessly or rewards immediate replay"),
    ("mobile_visuals", 0.5,
     "Made for a phone screen: vertical, subject centered and clear, on-screen text legible. Not production polish.",
     "hard to see what's happening on a phone; tiny or cluttered text",
     "instantly readable on a phone"),
    ("packaging", 0.5,
     "Title, description and hashtags: clear, intriguing, and matching the content.",
     "missing, spammy or misleading title",
     "concise title that adds intrigue and fits the video"),
]
KEYS = [c[0] for c in CATEGORIES]
WEIGHTS = {c[0]: c[1] for c in CATEGORIES}

SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "string"}} | {k: {"type": "integer"} for k in KEYS},
    "required": ["notes"] + KEYS,
}


def _rubric_text() -> str:
    lines = []
    for i, (k, _, d, lo, hi) in enumerate(CATEGORIES, 1):
        lines.append(f"{i}. {k}: {d}\n   1 = {lo}; 10 = {hi}")
    return "\n".join(lines)


PROMPT = f"""You are an expert YouTube Shorts analyst. Watch the Short and rate it on each \
category below from 1 to 10. Judge this specific video, not the creator. Use the full \
scale: 5 is an average Short in the feed. A strong video can still score low on some \
categories.

{_rubric_text()}

First write brief `notes` (2-4 sentences: what happens, and its main strengths and \
weaknesses), then give the integer scores."""


class Rubric:
    def __init__(self, model: str):
        self.llm = Gemini(model)

    def score(self, post: Post) -> dict:
        parts: list[dict] = []
        if url := video_url(post):
            parts.append(video_part(url))
        parts.append({"text": PROMPT + "\n\n" + post_text(post)})
        out = self.llm.run(post, parts, SCHEMA)
        if all(k in out for k in KEYS):
            out["score"] = weighted_raw(out)
        return out


def weighted_raw(r: dict, weights: dict[str, float] = WEIGHTS) -> float:
    """Weighted average of raw 1-10 scores, mapped to 0-100."""
    tot = sum(weights.values())
    avg = sum(weights[k] * r[k] for k in KEYS) / tot
    return round((avg - 1) / 9 * 100, 2)


def creator_ranked(results: list[dict], weights: dict[str, float] = WEIGHTS) -> dict[str, float]:
    """post_id -> 0-100 score from within-creator z-scores per category.

    For each creator and category, z = (score - creator mean) / creator sd
    (sd of 0 -> z = 0). Final = weighted mean z, mapped through a logistic
    to 0-100 so 50 means 'typical for this creator'."""
    import math

    by_creator: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if all(k in r for k in KEYS):
            by_creator[r["creator_id"]].append(r)
    out: dict[str, float] = {}
    tot = sum(weights.values())
    for rs in by_creator.values():
        stats = {k: (st.mean(r[k] for r in rs), st.pstdev([r[k] for r in rs])) for k in KEYS}
        for r in rs:
            z = sum(weights[k] * ((r[k] - stats[k][0]) / stats[k][1] if stats[k][1] else 0.0)
                    for k in KEYS) / tot
            out[r["post_id"]] = round(100 / (1 + math.exp(-1.5 * z)), 2)
    return out


def combined_rows(results: list[dict]) -> list[dict]:
    """Extra result rows, one per combination variant, labelled by model name."""
    equal = {k: 1.0 for k in KEYS}
    variants = {
        "rubric-ranked": creator_ranked(results),
        "rubric-equal": creator_ranked(results, equal),
    }
    rows = []
    for r in results:
        if "score" not in r:
            rows.append(r)  # keep errors visible in the report
            continue
        rows.append(r | {"model": f"{r['model']}|rubric-weighted"})
        for name, scores in variants.items():
            if r["post_id"] in scores:
                rows.append(r | {"model": f"{r['model']}|{name}", "score": scores[r["post_id"]]})
    return rows
