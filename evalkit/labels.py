"""Ground-truth labels: creator-relative engagement percentile.

Raw views mostly reflect audience size. Instead, each post is ranked against
the same creator's posts that are nearest in time (a rolling window), which
also absorbs follower growth over the account's history. 0 = worst post in the
window, 100 = best.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from .schema import Post


def engagement(post: Post, metric: str = "views", comment_weight: float = 2.0) -> float | None:
    """metric="views": log(1 + views), the main signal for Shorts, where reach
    is decided by how much the feed pushes the video.
    metric="interactions": log(1 + likes + comment_weight * comments)."""
    if metric == "views":
        views = post.hidden.get("views")
        return None if views is None else math.log1p(views)
    likes = post.hidden.get("likes")
    if likes is None:
        return None
    comments = post.hidden.get("comments") or 0
    return math.log1p(likes + comment_weight * comments)


def _midrank_pctile(value: float, others: list[float]) -> float:
    if not others:
        return 50.0
    below = sum(1 for o in others if o < value)
    ties = sum(1 for o in others if o == value)
    return 100.0 * (below + 0.5 * ties) / len(others)


def assign_creator_relative_labels(
    posts: Iterable[Post],
    window: int = 15,
    min_posts: int = 20,
    drop_most_recent: int = 0,
    metric: str = "views",
) -> list[Post]:
    """Return posts with hidden['label_raw'] and hidden['label_pctile'] set.

    window: compare against up to `window` posts before and after in time.
    min_posts: creators with fewer labelled posts are dropped.
    drop_most_recent: drop each creator's newest posts, for sources where
        immature posts weren't already filtered out by age at collection.
    """
    by_creator: dict[str, list[Post]] = defaultdict(list)
    for p in posts:
        e = engagement(p, metric)
        if e is None:
            continue
        p.hidden["label_raw"] = e
        by_creator[p.creator_id].append(p)

    out: list[Post] = []
    for plist in by_creator.values():
        plist.sort(key=lambda p: p.posted_at)
        if drop_most_recent:
            plist = plist[:-drop_most_recent]
        if len(plist) < min_posts:
            continue
        vals = [p.hidden["label_raw"] for p in plist]
        for i, p in enumerate(plist):
            lo, hi = max(0, i - window), min(len(plist), i + window + 1)
            others = vals[lo:i] + vals[i + 1 : hi]
            p.hidden["label_pctile"] = round(_midrank_pctile(vals[i], others), 2)
            out.append(p)
    return out
