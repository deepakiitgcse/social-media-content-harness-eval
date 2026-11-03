"""Run a scoring harness over a sample of posts.

Harnesses:
  default   one plain "rate 0-100" call per post (the baseline to beat)
  anchored  custom v1: shows K of the creator's earlier Shorts with real views
            (needs --history-file, e.g. dev_full.jsonl)
  rubric    custom v2: research-based category ratings, combined in code
            (see docs/shorts_research.md)

Usage:
    uv run python scripts/run_harness.py --harness default --split-file data/processed/youtube_2026-11-03/dev.jsonl \
        --n-creators 8 --per-creator 5 --models gemini-3.5-flash-lite gemini-3.8-flash gemini-4-argon \
        --run-name pilot
Writes runs/<run-name>/scores.jsonl (one line per model x post).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.loader import read_posts  # noqa: E402
from evalkit.models.gemini import Gemini  # noqa: E402
from evalkit.splits import sample_eval_set  # noqa: E402
from harness.anchored import Anchored  # noqa: E402
from harness.rubric import Rubric, combined_rows  # noqa: E402

DEFAULT_PROMPT = (
    "You are a social media expert. Rate the quality of this YouTube Short "
    "from 0 to 100, where 100 is excellent. Return JSON with `score` and a "
    "one-sentence `reason`."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--n-creators", type=int, default=8)
    ap.add_argument("--per-creator", type=int, default=5)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--run-name", default="pilot")
    ap.add_argument("--harness", choices=["default", "anchored", "rubric"], default="default")
    ap.add_argument("--history-file", type=Path)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    posts = sample_eval_set(read_posts(a.split_file), a.n_creators, a.per_creator, a.seed)
    out_dir = Path("runs") / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        vars(a) | {"split_file": str(a.split_file), "history_file": str(a.history_file),
                   "prompt": DEFAULT_PROMPT if a.harness == "default" else f"harness/{a.harness}.py",
                   "post_ids": [p.post_id for p in posts]}, indent=2))

    if a.harness == "default":
        scorers = [lambda p, g=Gemini(m): g.score(p, DEFAULT_PROMPT) for m in a.models]
    elif a.harness == "rubric":
        scorers = [Rubric(m).score for m in a.models]
    else:
        if not a.history_file:
            ap.error("--history-file is required for the anchored harness")
        history = list(read_posts(a.history_file))
        scorers = [Anchored(m, history, k=a.k).score for m in a.models]
    jobs = [(s, p) for s in scorers for p in posts]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda j: j[0](j[1]), jobs))

    by_id = {p.post_id: p for p in posts}
    for r in results:
        p = by_id[r["post_id"]]
        r |= {"creator_id": p.creator_id, "label_pctile": p.hidden["label_pctile"],
              "views": p.hidden["views"], "title": p.title}
    # Rubric scores are combined across posts (creator ranking), so rows are
    # produced after all posts are scored, one per combination variant.
    rows = combined_rows(results) if a.harness == "rubric" else results
    with open(out_dir / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errs = [r for r in results if "error" in r]
    print(f"{len(results) - len(errs)}/{len(results)} scored -> {out_dir}/scores.jsonl")
    for r in errs[:5]:
        print("error:", r["model"], r["post_id"], str(r["error"])[:200])


if __name__ == "__main__":
    main()
