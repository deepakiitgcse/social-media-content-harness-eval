"""Read/write posts.jsonl and guard against engagement leakage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import HIDDEN_KEYS, Post


def read_posts(path: str | Path) -> Iterator[Post]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Post.from_dict(json.loads(line))


def write_posts(posts: Iterable[Post], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for p in posts:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


class LeakageError(RuntimeError):
    pass


def assert_no_leak(payload: Any, post: Post | None = None) -> None:
    """Raise if a model payload contains hidden keys or this post's exact
    like/comment counts as numbers. Call it on every prompt before sending."""
    def walk(x: Any) -> Iterator[tuple[str | None, Any]]:
        if isinstance(x, dict):
            for k, v in x.items():
                yield k, v
                yield from walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                yield None, v
                yield from walk(v)

    for k, _ in walk(payload):
        if k in HIDDEN_KEYS:
            raise LeakageError(f"hidden key {k!r} in model payload")

    if post is not None:
        text = json.dumps(payload, ensure_ascii=False)
        for key in ("likes", "comments", "views"):
            v = post.hidden.get(key)
            # Small counts collide with ordinary numbers; only check distinctive ones.
            if isinstance(v, int) and v >= 1000 and str(v) in text:
                raise LeakageError(f"post's {key} count {v} appears in model payload")
