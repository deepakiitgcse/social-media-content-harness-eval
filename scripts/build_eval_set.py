"""Label posts, assign creator-level splits, and sample the eval set.

Usage:
    uv run python scripts/build_eval_set.py --in data/raw/youtube/<date>/posts_all.jsonl \
        --out-dir data/processed/youtube_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.labels import assign_creator_relative_labels  # noqa: E402
from evalkit.loader import read_posts, write_posts  # noqa: E402
from evalkit.splits import SPLITS, sample_eval_set, split_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--creators-per-split", type=int, default=100)
    ap.add_argument("--posts-per-creator", type=int, default=10)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--min-posts", type=int, default=20)
    ap.add_argument("--drop-most-recent", type=int, default=0)
    ap.add_argument("--metric", choices=["views", "interactions"], default="views")
    ap.add_argument("--keep-sponsored", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    posts = (p for p in read_posts(a.inp) if a.keep_sponsored or not p.is_sponsored)
    labelled = assign_creator_relative_labels(
        posts, window=a.window, min_posts=a.min_posts,
        drop_most_recent=a.drop_most_recent, metric=a.metric,
    )
    by_split = {name: [] for name, _ in SPLITS}
    for p in labelled:
        by_split[split_of(p.creator_id)].append(p)

    summary = {}
    for name, posts in by_split.items():
        # The full labelled train split is kept for retrieval / calibration;
        # dev and test are sampled down to keep model costs bounded.
        write_posts(posts, a.out_dir / f"{name}_full.jsonl")
        sample = sample_eval_set(posts, a.creators_per_split, a.posts_per_creator, a.seed)
        write_posts(sample, a.out_dir / f"{name}.jsonl")
        summary[name] = {
            "creators_available": len({p.creator_id for p in posts}),
            "posts_available": len(posts),
            "eval_posts": len(sample),
            "eval_creators": len({p.creator_id for p in sample}),
            "niches": Counter(p.creator_niche for p in sample).most_common(10),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
