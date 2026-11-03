import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.labels import assign_creator_relative_labels
from evalkit.loader import LeakageError, assert_no_leak, read_posts, write_posts
from evalkit.schema import HIDDEN_KEYS, Media, Post
from evalkit.splits import sample_eval_set, split_of


def make_posts(n_creators=6, n_posts=40, seed=0):
    rng = random.Random(seed)
    posts = []
    for c in range(n_creators):
        base = 10 ** rng.uniform(2, 6)  # very different audience sizes
        for i in range(n_posts):
            posts.append(Post(
                post_id=f"c{c}p{i}", platform="instagram", creator_id=f"creator{c}",
                posted_at=f"2019-01-{1 + i // 2:02d}T{(i % 2) * 12:02d}:00:00+00:00",
                media=[Media("image", f"x/{c}_{i}.jpg")], caption=f"post {i} #fit",
                hashtags=["fit"],
                hidden={"views": int(base * rng.uniform(0.3, 3)), "likes": int(base * rng.uniform(0.01, 0.1)),
                        "comments": rng.randint(0, 50),
                        "followers": int(base * 30)},
            ))
    return posts


def test_public_view_has_no_hidden_fields():
    p = make_posts(1, 1)[0]
    view = p.public_view()
    assert not HIDDEN_KEYS & set(view)
    assert "creator_id" not in view
    assert_no_leak(view, p)


def test_leak_detection():
    p = make_posts(1, 1)[0]
    p.hidden["views"] = 123456
    with pytest.raises(LeakageError):
        assert_no_leak({"followers": 10})
    with pytest.raises(LeakageError):
        assert_no_leak({"text": "this got 123456 views"}, p)


def test_labels_are_creator_relative_and_uniformish():
    posts = assign_creator_relative_labels(make_posts(), window=15, min_posts=20, drop_most_recent=2)
    assert len(posts) == 6 * 38
    labels = [p.hidden["label_pctile"] for p in posts]
    assert all(0 <= v <= 100 for v in labels)
    # Every creator spans the range regardless of audience size.
    for c in {p.creator_id for p in posts}:
        vals = [p.hidden["label_pctile"] for p in posts if p.creator_id == c]
        assert min(vals) < 15 and max(vals) > 85


def test_min_posts_filter():
    assert assign_creator_relative_labels(make_posts(2, 10), min_posts=20) == []


def test_splits_stable_and_by_creator():
    assert split_of("abc") == split_of("abc")
    names = [split_of(f"creator{i}") for i in range(2000)]
    frac = {n: names.count(n) / len(names) for n in set(names)}
    assert abs(frac["train"] - 0.5) < 0.05 and abs(frac["test"] - 0.25) < 0.05


def test_sample_eval_set():
    posts = assign_creator_relative_labels(make_posts())
    s = sample_eval_set(posts, n_creators=4, posts_per_creator=10)
    assert len(s) == 40 and len({p.creator_id for p in s}) == 4


def test_roundtrip(tmp_path):
    posts = make_posts(1, 3)
    write_posts(posts, tmp_path / "p.jsonl")
    assert [p.to_dict() for p in read_posts(tmp_path / "p.jsonl")] == [p.to_dict() for p in posts]
