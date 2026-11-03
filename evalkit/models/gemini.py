"""Gemini adapter (REST generateContent, API key from GEMINI_API_KEY).

The Short itself goes in as a YouTube URL, which Gemini fetches and watches
natively; the post's public text fields go in the prompt. Responses are
cached on disk by (model, request) so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..env import load_env
from ..loader import assert_no_leak
from ..schema import Post

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}


def post_text(post: Post) -> str:
    v = post.public_view()
    lines = [
        f"Platform: YouTube Shorts",
        f"Title: {v['title']}",
        f"Description: {v['caption'] or '(none)'}",
        f"Hashtags: {', '.join('#' + h for h in v['hashtags']) or '(none)'}",
        f"Duration: {v['duration_s']:.0f} seconds" if v["duration_s"] else "",
        f"Category: {v['creator_niche'] or 'unknown'}",
    ]
    return "\n".join(l for l in lines if l)


def video_url(post: Post) -> str | None:
    for m in post.media:
        if m.type == "video" and m.url:
            # Plain watch URLs are the documented form for Gemini's YouTube input.
            return m.url.replace("/shorts/", "/watch?v=")
    return None


class Gemini:
    def __init__(self, model: str, cache_dir: Path = Path("runs/cache/gemini")):
        load_env()
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY not set (.env)")
        self.model = model
        self.cache_dir = cache_dir / model
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        ck = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        cpath = self.cache_dir / f"{ck}.json"
        if cpath.exists():
            return json.loads(cpath.read_text())
        req = urllib.request.Request(
            API.format(model=self.model), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
        )
        last = None
        for attempt in range(5):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.load(r)
                data["_latency_s"] = round(time.time() - t0, 2)
                cpath.write_text(json.dumps(data))
                return data
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                last = f"HTTP {e.code}: {body[:300]}"
                # Per-day quota won't recover within our retry window; retrying just burns requests.
                # Empty-body 404s showed up transiently while billing was being enabled.
                transient = e.code in (429, 500, 502, 503, 504) or (e.code == 404 and not body.strip())
                if not transient or "PerDay" in body:
                    if "PerDay" in body:
                        last = f"HTTP {e.code}: daily quota exhausted for {self.model}"
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                last = str(getattr(e, "reason", e))
            time.sleep(2 ** attempt * 2)
        return {"error": last}

    def score(self, post: Post, prompt: str) -> dict[str, Any]:
        parts: list[dict] = []
        url = video_url(post)
        if url:
            parts.append(video_part(url))
        parts.append({"text": prompt + "\n\n" + post_text(post)})
        return self.run(post, parts, SCORE_SCHEMA)

    def run(self, post: Post, parts: list[dict], schema: dict) -> dict[str, Any]:
        """Send `parts` with a JSON response schema. Returns the parsed fields
        plus usage, or an `error`."""
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
        }
        assert_no_leak(body, post)
        data = self.generate(body)
        out: dict[str, Any] = {"model": self.model, "post_id": post.post_id,
                               "usage": data.get("usageMetadata"), "latency_s": data.get("_latency_s")}
        if "error" in data:
            return out | {"error": data["error"]}
        try:
            raw = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            parsed = json.loads(raw)
            if "score" in parsed:
                parsed["score"] = float(parsed["score"])
            return out | parsed
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return out | {"error": f"unparseable response: {e}", "raw": json.dumps(data)[:500]}


def video_part(url: str) -> dict:
    return {"file_data": {"file_uri": url, "mime_type": "video/mp4"}}
