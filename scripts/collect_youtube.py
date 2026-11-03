"""Collect YouTube Shorts from small creators via the official Data API v3.

Steps:
  1. Discover channels: search.list for recent short videos across niche
     queries (100 quota units per page, the expensive part).
  2. Keep small creators: subscriber count within [--min-subs, --max-subs].
  3. For each channel, list its Shorts published in the window
     (Shorts-only playlist, falling back to uploads + duration/aspect checks).
  4. Fetch video details/stats and keep channels with >= --min-shorts Shorts.

Only posts at least --min-age-days old are kept, so view counts have mostly
settled. Default window starts after current models' training cutoffs.

Usage:
    export YOUTUBE_API_KEY=...
    uv run python scripts/collect_youtube.py            # writes data/raw/youtube/<date>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.loader import write_posts  # noqa: E402
from evalkit.youtube import (  # noqa: E402
    QuotaExceeded, YouTubeAPI, shorts_playlist_id, uploads_playlist_id, video_to_post,
)

DEFAULT_QUERIES = [
    "cooking", "recipe", "fitness", "workout", "comedy skit", "gaming", "minecraft",
    "diy", "crafts", "makeup", "fashion", "pets", "dog", "cat", "travel", "tech review",
    "science", "study tips", "personal finance", "art", "drawing", "music cover",
    "guitar", "football", "basketball", "car", "gardening", "life hacks", "motivation",
    "home decor",
]


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("data/raw/youtube"))
    ap.add_argument("--published-after", default="2026-07-01T00:00:00Z")
    ap.add_argument("--min-age-days", type=int, default=14)
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--search-pages", type=int, default=1)
    ap.add_argument("--min-subs", type=int, default=1_000)
    ap.add_argument("--max-subs", type=int, default=100_000)
    ap.add_argument("--min-shorts", type=int, default=25)
    ap.add_argument("--max-channels", type=int, default=400)
    ap.add_argument("--max-quota", type=int, default=9000)
    a = ap.parse_args()

    now = datetime.now(timezone.utc)
    snap = a.out_root / now.strftime("%Y-%m-%d")
    api = YouTubeAPI(snap / "api_cache", max_quota=a.max_quota)
    # Day resolution, so re-runs on the same day hit the response cache.
    published_before = (now - timedelta(days=a.min_age_days)).strftime("%Y-%m-%dT00:00:00Z")
    stats: dict[str, int] = {}

    # 1. Discovery
    channel_ids: dict[str, str] = {}
    try:
        for q in a.queries:
            for it in api.search_short_videos(q, a.published_after, published_before, a.search_pages):
                channel_ids.setdefault(it["snippet"]["channelId"], q)
    except QuotaExceeded as e:
        print(f"discovery stopped early: {e}", file=sys.stderr)
    stats["channels_discovered"] = len(channel_ids)

    # 2. Small creators only
    small = []
    for ch in api.channels(list(channel_ids)):
        st = ch.get("statistics", {})
        if st.get("hiddenSubscriberCount"):
            continue
        if a.min_subs <= int(st.get("subscriberCount", 0)) <= a.max_subs:
            ch["_discovery_query"] = channel_ids[ch["id"]]
            small.append(ch)
    small = small[: a.max_channels]
    stats["channels_small"] = len(small)

    # 3–4. Shorts per channel
    posts, kept_channels = [], []
    try:
        for ch in small:
            ids = api.playlist_video_ids(shorts_playlist_id(ch["id"]), a.published_after)
            require_vertical = ids is None
            if ids is None:
                ids = api.playlist_video_ids(uploads_playlist_id(ch["id"]), a.published_after) or []
            if len(ids) < a.min_shorts:
                continue
            ch_posts = []
            for v in api.videos(ids):
                if v["snippet"]["publishedAt"] > published_before:
                    continue  # too recent; views not settled
                p = video_to_post(v, ch, require_vertical=require_vertical)
                if p:
                    ch_posts.append(p)
            if len(ch_posts) >= a.min_shorts:
                posts += ch_posts
                kept_channels.append(ch)
    except QuotaExceeded as e:
        print(f"collection stopped early: {e}", file=sys.stderr)

    stats.update(channels_kept=len(kept_channels), posts=len(posts), quota_used=api.quota_used)
    write_posts(posts, snap / "posts_all.jsonl")
    with open(snap / "channels.jsonl", "w") as f:
        for ch in kept_channels:
            f.write(json.dumps(ch) + "\n")
    (snap / "meta.json").write_text(json.dumps({
        "collected_at": iso(now), "published_after": a.published_after,
        "published_before": published_before, "args": vars(a) | {"out_root": str(a.out_root)},
        "stats": stats,
        "delete_or_refresh_by": iso(now + timedelta(days=30)),
    }, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"snapshot: {snap}  (YouTube API policy: delete or refresh within 30 days)")


if __name__ == "__main__":
    main()
