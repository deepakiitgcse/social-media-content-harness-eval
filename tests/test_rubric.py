import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.rubric import KEYS, combined_rows, creator_ranked, weighted_raw


def row(pid, creator, val, **over):
    return {"model": "m", "post_id": pid, "creator_id": creator, "label_pctile": 50.0,
            **{k: val for k in KEYS}, **over} | {"score": 0.0}


def test_weighted_raw_bounds():
    assert weighted_raw({k: 1 for k in KEYS}) == 0
    assert weighted_raw({k: 10 for k in KEYS}) == 100


def test_creator_ranking_removes_channel_level():
    # Creator A is uniformly "better" than B, but within each creator the
    # ordering is the same, so ranked scores should match across creators.
    rs = [row("a1", "A", 9), row("a2", "A", 7), row("b1", "B", 4), row("b2", "B", 2)]
    s = creator_ranked(rs)
    assert s["a1"] == s["b1"] > 50 > s["a2"] == s["b2"]


def test_zero_variance_category_is_neutral():
    rs = [row("a1", "A", 5), row("a2", "A", 5)]
    assert creator_ranked(rs) == {"a1": 50.0, "a2": 50.0}


def test_combined_rows_variants_and_errors():
    rs = [row("a1", "A", 9), row("a2", "A", 7), {"model": "m", "post_id": "x", "error": "boom"}]
    models = {r["model"] for r in combined_rows(rs)}
    assert models == {"m", "m|rubric-weighted", "m|rubric-ranked", "m|rubric-equal"}
