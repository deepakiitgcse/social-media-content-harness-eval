"""Minimal YouTube Data API v3 client (official API only, API key auth).

Responses are cached on disk under the snapshot directory, so re-runs don't
spend quota. The API key is read from YOUTUBE_API_KEY and never written to
the cache or logs.

YouTube API policy: statistics fetched without the channel owner's
authorization may be stored for at most 30 days. Each collection run writes a
dated snapshot; scripts/purge_youtube.py deletes snapshots past that limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "https://www.googleapis.com/youtube/v3/"
# Quota cost per call (https://developers.google.com/youtube/v3/determine_quota_cost)
COST = {"search": 100, "channels": 1, "playlistItems": 1, "videos": 1}

# Standard YouTube video categories (videoCategories.list, US).
CATEGORIES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals",
    "17": "Sports", "19": "Travel & Events", "20": "Gaming", "22": "People & Blogs",
    "23": "Comedy", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
}


class QuotaExceeded(RuntimeError):
    pass


class YouTubeAPI:
    def __init__(self, cache_dir: Path, api_key: str | None = None, max_quota: int = 9000):
        self.key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not self.key:
            raise RuntimeError("Set YOUTUBE_API_KEY (Google Cloud console → YouTube Data API v3).")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_quota = max_quota
        self.quota_used = 0

    def get(self, endpoint: str, **params: Any) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        ck = hashlib.sha256(f"{endpoint}?{sorted(params.items())}".encode()).hexdigest()[:24]
        cpath = self.cache_dir / f"{endpoint}_{ck}.json"
        if cpath.exists():
            return json.loads(cpath.read_text())
        cost = COST[endpoint]
        if self.quota_used + cost > self.max_quota:
            raise QuotaExceeded(f"would exceed max_quota={self.max_quota}")
        url = API + endpoint + "?" + urllib.parse.urlencode({**params, "key": self.key})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if "quotaExceeded" in body:
                    raise QuotaExceeded("daily quota exhausted") from None
                data = {"error": {"code": e.code, "body": body[:500]}}
                if e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 4:
                    # Re-runs resume from the cache, so failing loudly is cheap.
                    raise RuntimeError(f"{endpoint}: network error after retries: "
                                       f"{getattr(e, 'reason', e)}") from None
            time.sleep(2 ** attempt)
        self.quota_used += cost
        if "error" not in data or data["error"].get("code") == 404:
            cpath.write_text(json.dumps(data))
        return data

    # --- endpoints -----------------------------------------------------------

    def search_short_videos(self, q: str, published_after: str, published_before: str,
                            pages: int = 1, region: str = "US", lang: str = "en") -> list[dict]:
        items, token = [], None
        for _ in range(pages):
            d = self.get("search", part="snippet", type="video", videoDuration="short",
                         q=q, order="date", maxResults=50, publishedAfter=published_after,
                         publishedBefore=published_before, regionCode=region,
                         relevanceLanguage=lang, pageToken=token)
            items += d.get("items", [])
            token = d.get("nextPageToken")
            if not token:
                break
        return items

    def channels(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            d = self.get("channels", part="snippet,statistics,contentDetails",
                         id=",".join(ids[i:i + 50]), maxResults=50)
            out += d.get("items", [])
        return out

    def playlist_video_ids(self, playlist_id: str, published_after: str,
                           max_pages: int = 6) -> list[str] | None:
        """Video ids in a playlist, newest first, stopping once older than
        published_after. Returns None if the playlist doesn't exist."""
        ids, token = [], None
        for _ in range(max_pages):
            d = self.get("playlistItems", part="contentDetails", playlistId=playlist_id,
                         maxResults=50, pageToken=token)
            if "error" in d:
                return None
            for it in d.get("items", []):
                cd = it["contentDetails"]
                if cd.get("videoPublishedAt", "9999") < published_after:
                    return ids
                ids.append(cd["videoId"])
            token = d.get("nextPageToken")
            if not token:
                break
        return ids

    def videos(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            # maxWidth makes the player part return embed dimensions (aspect ratio).
            d = self.get("videos", part="snippet,contentDetails,statistics,player",
                         id=",".join(ids[i:i + 50]), maxWidth=360, maxResults=50)
            out += d.get("items", [])
        return out


# --- helpers -----------------------------------------------------------------

def shorts_playlist_id(channel_id: str) -> str:
    """Shorts-only uploads playlist ("UUSH" + id without "UC"). Undocumented
    but served by the official playlistItems endpoint; callers fall back to
    the regular uploads playlist plus duration/aspect checks if it's missing."""
    return "UUSH" + channel_id[2:]


def uploads_playlist_id(channel_id: str) -> str:
    return "UU" + channel_id[2:]


def parse_iso_duration(s: str) -> float | None:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", s or "")
    if not m:
        return None
    d, h, mi, sec = m.groups()
    return int(d or 0) * 86400 + int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0)


def is_vertical(video: dict) -> bool | None:
    p = video.get("player", {})
    w, h = p.get("embedWidth"), p.get("embedHeight")
    if not w or not h:
        return None
    return int(h) > int(w)


def best_thumbnail_url(video: dict) -> str | None:
    th = video.get("snippet", {}).get("thumbnails", {})
    for k in ("maxres", "standard", "high", "medium", "default"):
        if k in th:
            return th[k]["url"]
    return None


def video_to_post(video: dict, channel: dict, max_duration_s: float = 180.0,
                  require_vertical: bool = False):
    """Convert API video + channel items to a Post, or None if it isn't a
    usable Short (too long, live, no view count, or not vertical when the
    caller couldn't use the Shorts-only playlist)."""
    from .schema import Media, Post

    sn, cd, st = video.get("snippet", {}), video.get("contentDetails", {}), video.get("statistics", {})
    dur = parse_iso_duration(cd.get("duration", ""))
    if dur is None or dur > max_duration_s or "viewCount" not in st:
        return None
    if sn.get("liveBroadcastContent", "none") != "none":
        return None
    vertical = is_vertical(video)
    if require_vertical and vertical is False:
        return None

    title, desc = sn.get("title", ""), sn.get("description", "")
    hashtags = [h.lower() for h in re.findall(r"#(\w+)", f"{title} {desc}")]
    cst = channel.get("statistics", {})
    hidden = {
        "views": int(st["viewCount"]),
        "likes": int(st["likeCount"]) if "likeCount" in st else None,
        "comments": int(st["commentCount"]) if "commentCount" in st else None,
        "followers": None if cst.get("hiddenSubscriberCount") else int(cst.get("subscriberCount", 0)),
        "thumbnail_url": best_thumbnail_url(video),
        "is_vertical": vertical,
    }
    return Post(
        post_id=video["id"],
        platform="youtube_shorts",
        creator_id=sn["channelId"],
        posted_at=sn["publishedAt"],
        media=[Media("video", url=f"https://www.youtube.com/shorts/{video['id']}")],
        title=title,
        caption=desc,
        hashtags=hashtags,
        duration_s=dur,
        is_sponsored=bool(video.get("paidProductPlacementDetails", {}).get("hasPaidProductPlacement")),
        creator_niche=CATEGORIES.get(sn.get("categoryId", ""), sn.get("categoryId")),
        hidden=hidden,
    )
