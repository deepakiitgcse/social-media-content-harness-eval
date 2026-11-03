import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evalkit.loader import assert_no_leak
from evalkit.schema import Media, Post
from harness.anchored import Anchored


def post(i, views, creator="UC1"):
    return Post(post_id=f"p{i}", platform="youtube_shorts", creator_id=creator,
                posted_at=f"2026-08-{1 + i // 4:02d}T{(i % 4) * 6:02d}:00:00Z",
                media=[Media("video", url=f"https://www.youtube.com/shorts/p{i}")],
                title=f"short {i}", hidden={"views": views})


class FakeLLM:
    def __init__(self):
        self.calls = []

    def run(self, target, parts, schema):
        assert_no_leak({"parts": parts}, target)
        self.calls.append(parts)
        return {"post_id": target.post_id, "score": 50.0}


def make(history, k=4):
    h = Anchored.__new__(Anchored)
    h.llm, h.k, h.lookback = FakeLLM(), k, 30
    h.by_creator = {}
    for p in history:
        h.by_creator.setdefault(p.creator_id, []).append(p)
    for ps in h.by_creator.values():
        ps.sort(key=lambda p: p.posted_at)
    return h


def test_references_are_earlier_same_creator_and_span_range():
    hist = [post(i, views=100 * (i + 1)) for i in range(40)] + [post(99, 5, creator="UC2")]
    target = hist[35]
    refs = make(hist).references(target)
    assert len(refs) == 4
    assert all(r.creator_id == "UC1" and r.posted_at < target.posted_at for r in refs)
    views = [r.hidden["views"] for r in refs]
    assert views == sorted(views) and views[-1] / views[0] > 3


def test_tops_up_with_later_posts_when_history_is_short():
    hist = [post(i, views=1000 + i) for i in range(6)]
    refs = make(hist).references(hist[1])
    assert len(refs) == 4 and "p1" not in {r.post_id for r in refs}


def test_target_views_never_in_prompt():
    hist = [post(i, views=12345 + i * 1000) for i in range(10)]
    target = hist[9]
    h = make(hist)
    out = h.score(target)  # FakeLLM asserts no leak
    text = " ".join(p.get("text", "") for p in h.llm.calls[0])
    assert f"{target.hidden['views']:,}" not in text and str(target.hidden["views"]) not in text
    assert "NEW Short" in text and len(out["reference_ids"]) == 4
