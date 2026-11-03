"""Custom harness v1: creator-anchored ranking.

The pilot showed default prompts rate the *channel* (scores vary ~3x more
between creators than within one), while the label asks how a Short does
relative to the creator's own typical Short. So this harness shows the model
K of the same creator's earlier Shorts with their real view counts, spanning
low to high, and asks where the new Short will land among them.

This is information a creator has in the product (their own past results).
Only other posts' views are shown; the target's never is (assert_no_leak).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from evalkit.models.gemini import Gemini, post_text, video_part, video_url
from evalkit.schema import Post

SCHEMA = {
    "type": "object",
    "properties": {
        "comparison": {"type": "string"},
        "hook": {"type": "integer"},
        "visual_quality": {"type": "integer"},
        "clarity": {"type": "integer"},
        "novelty": {"type": "integer"},
        "title_and_packaging": {"type": "integer"},
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["comparison", "hook", "visual_quality", "clarity", "novelty",
                 "title_and_packaging", "score", "reason"],
}

PROMPT = """You are predicting how well a creator's new YouTube Short will perform \
compared with that same creator's other Shorts. Audience size is the same for all of \
them, so differences in views come from the content itself: the hook in the first \
seconds, pacing, clarity, novelty, emotional pull, and title/packaging.

Below are {k} earlier Shorts from this creator with their real view counts, from \
fewest to most views. Study what separates the stronger ones from the weaker ones.

{refs}

Then watch the NEW Short (the last video) and:
1. In `comparison`, briefly say which references it most resembles in quality and why.
2. Rate hook, visual_quality, clarity, novelty, title_and_packaging from 1-10.
3. Give `score`: the percentile (0-100) of this creator's Shorts that the new Short \
will outperform. 50 = typical for this creator, 90 = one of their best, 10 = one of \
their weakest. Use the full range when warranted.
4. `reason`: one sentence."""


def fmt_views(n: int) -> str:
    return f"{n:,}"


class Anchored:
    def __init__(self, model: str, history: Iterable[Post], k: int = 4, lookback: int = 30):
        self.llm = Gemini(model)
        self.k = k
        self.lookback = lookback
        self.by_creator: dict[str, list[Post]] = defaultdict(list)
        for p in history:
            self.by_creator[p.creator_id].append(p)
        for ps in self.by_creator.values():
            ps.sort(key=lambda p: p.posted_at)

    def references(self, target: Post) -> list[Post]:
        """K posts by the same creator spanning their view range. Prefers the
        `lookback` most recent posts before the target; tops up with the
        nearest later posts only if there aren't enough earlier ones."""
        ps = [p for p in self.by_creator[target.creator_id] if p.post_id != target.post_id]
        before = [p for p in ps if p.posted_at < target.posted_at][-self.lookback:]
        if len(before) < self.k:
            after = [p for p in ps if p.posted_at >= target.posted_at]
            before += after[: self.k - len(before)]
        pool = sorted(before, key=lambda p: p.hidden["views"])
        if len(pool) <= self.k:
            return pool
        # Evenly spaced quantiles, e.g. k=4 -> 10th, 37th, 63rd, 90th percentile.
        idx = [round((0.1 + 0.8 * i / (self.k - 1)) * (len(pool) - 1)) for i in range(self.k)]
        return [pool[i] for i in sorted(set(idx))]

    def score(self, target: Post) -> dict:
        refs = self.references(target)
        parts: list[dict] = []
        ref_lines = []
        for i, r in enumerate(refs, 1):
            ref_lines.append(f"Reference {i}: {fmt_views(r.hidden['views'])} views")
        parts.append({"text": PROMPT.format(k=len(refs), refs="\n".join(ref_lines))})
        for i, r in enumerate(refs, 1):
            parts.append({"text": f"\n--- Reference {i} ({fmt_views(r.hidden['views'])} views) ---\n"
                                  + post_text(r)})
            if url := video_url(r):
                parts.append(video_part(url))
        parts.append({"text": "\n--- NEW Short (predict this one) ---\n" + post_text(target)})
        if url := video_url(target):
            parts.append(video_part(url))
        out = self.llm.run(target, parts, SCHEMA)
        out["reference_ids"] = [r.post_id for r in refs]
        return out
