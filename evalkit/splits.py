"""Creator-level splits and eval-set sampling.

Splits are by creator so no creator's posts appear in more than one split.
Assignment is a stable hash of creator_id, so it never changes between runs.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable

from .schema import Post

SPLITS = (("train", 0.50), ("dev", 0.25), ("test", 0.25))


def split_of(creator_id: str, salt: str = "v1") -> str:
    h = int(hashlib.sha256(f"{salt}:{creator_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return SPLITS[-1][0]


def sample_eval_set(
    posts: Iterable[Post],
    n_creators: int,
    posts_per_creator: int,
    seed: int = 0,
) -> list[Post]:
    """Pick n_creators at random, then posts_per_creator posts from each.

    Creator-relative percentiles are roughly uniform within each creator, so
    uniform sampling within a creator gives a balanced label distribution.
    """
    rng = random.Random(seed)
    by_creator: dict[str, list[Post]] = defaultdict(list)
    for p in posts:
        by_creator[p.creator_id].append(p)
    eligible = sorted(c for c, ps in by_creator.items() if len(ps) >= posts_per_creator)
    chosen = rng.sample(eligible, min(n_creators, len(eligible)))
    out: list[Post] = []
    for c in chosen:
        out.extend(rng.sample(by_creator[c], posts_per_creator))
    return out
