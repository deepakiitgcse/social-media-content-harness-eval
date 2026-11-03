import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evalkit.loader import assert_no_leak, read_posts
from evalkit.youtube import YouTubeAPI, parse_iso_duration, video_to_post


def video(vid, ch="UCabc", published="2026-08-01T12:00:00Z", dur="PT45S", views=1000,
          w=203, h=360):
    return {
        "id": vid,
        "snippet": {"channelId": ch, "publishedAt": published, "title": f"Crispy tofu #shorts {vid}",
                    "description": "3 ingredients #vegan", "categoryId": "26",
                    "liveBroadcastContent": "none",
                    "thumbnails": {"high": {"url": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}}},
        "contentDetails": {"duration": dur},
        "statistics": {"viewCount": str(views), "likeCount": "40", "commentCount": "3"},
        "player": {"embedWidth": str(w), "embedHeight": str(h)},
    }


CHANNEL = {"id": "UCabc", "statistics": {"subscriberCount": "12000", "hiddenSubscriberCount": False}}


@pytest.mark.parametrize("s,expected", [
    ("PT45S", 45), ("PT1M2S", 62), ("PT3M", 180), ("PT1H", 3600), ("P0D", 0), ("bad", None),
])
def test_parse_iso_duration(s, expected):
    assert parse_iso_duration(s) == expected


def test_video_to_post():
    p = video_to_post(video("v1", views=98765), CHANNEL)
    assert p.platform == "youtube_shorts" and p.creator_id == "UCabc"
    assert p.duration_s == 45 and p.creator_niche == "Howto & Style"
    assert p.hashtags == ["shorts", "vegan"]
    assert p.hidden["views"] == 98765 and p.hidden["followers"] == 12000
    assert p.media[0].url == "https://www.youtube.com/shorts/v1"
    assert_no_leak(p.public_view(), p)


def test_video_filters():
    assert video_to_post(video("v", dur="PT4M"), CHANNEL) is None
    landscape = video("v", w=640, h=360)
    assert video_to_post(landscape, CHANNEL) is not None
    assert video_to_post(landscape, CHANNEL, require_vertical=True) is None
    no_views = video("v")
    del no_views["statistics"]["viewCount"]
    assert video_to_post(no_views, CHANNEL) is None


class FakeAPI(YouTubeAPI):
    """Serves canned responses instead of calling the network."""

    def __init__(self, cache_dir, max_quota=9000, **_):
        super().__init__(cache_dir, api_key="test", max_quota=max_quota)
        self.calls = []

    def get(self, endpoint, **params):
        self.calls.append(endpoint)
        if endpoint == "search":
            return {"items": [{"snippet": {"channelId": c}} for c in ("UCsmall", "UCbig")]}
        if endpoint == "channels":
            return {"items": [
                {"id": "UCsmall", "statistics": {"subscriberCount": "5000"}},
                {"id": "UCbig", "statistics": {"subscriberCount": "9000000"}},
            ]}
        if endpoint == "playlistItems":
            assert params["playlistId"] == "UUSHsmall"
            return {"items": [{"contentDetails": {"videoId": f"s{i}",
                                                  "videoPublishedAt": "2026-08-01T00:00:00Z"}}
                              for i in range(30)]}
        if endpoint == "videos":
            return {"items": [video(i, ch="UCsmall", views=100 + n)
                              for n, i in enumerate(params["id"].split(","))]}
        raise AssertionError(endpoint)


def test_collect_end_to_end(tmp_path, monkeypatch):
    import collect_youtube
    monkeypatch.setattr(collect_youtube, "YouTubeAPI", FakeAPI)
    monkeypatch.setattr(sys, "argv", ["x", "--out-root", str(tmp_path), "--queries", "cooking",
                                      "--min-shorts", "25"])
    collect_youtube.main()
    snap = next(tmp_path.iterdir())
    posts = list(read_posts(snap / "posts_all.jsonl"))
    assert len(posts) == 30 and {p.creator_id for p in posts} == {"UCsmall"}
    meta = json.loads((snap / "meta.json").read_text())
    assert meta["stats"]["channels_small"] == 1
    # The API key must not end up on disk.
    assert all('"test"' not in f.read_text() for f in snap.rglob("*") if f.is_file())
