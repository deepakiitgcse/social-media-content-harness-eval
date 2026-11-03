"""Post record schema.

A Post has public fields (what a model may see) and a `hidden` dict holding
engagement outcomes and labels. Only `Post.public_view()` may be passed to a
model; the loader and tests enforce this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Keys that must never appear in anything sent to a model.
HIDDEN_KEYS = frozenset(
    {"likes", "comments", "views", "followers", "following", "label_pctile", "label_raw",
     "viewCount", "likeCount", "commentCount", "subscriberCount"}
)


@dataclass
class Media:
    type: str  # "image" | "video"
    path: str | None = None  # local file, relative to data root
    url: str | None = None  # e.g. public YouTube URL (Gemini can read these natively)


@dataclass
class Post:
    post_id: str
    platform: str
    creator_id: str
    posted_at: str  # ISO 8601 UTC
    media: list[Media]
    title: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    duration_s: float | None = None
    is_sponsored: bool | None = None
    creator_niche: str | None = None
    hidden: dict[str, Any] = field(default_factory=dict)

    def public_view(self) -> dict[str, Any]:
        """Fields a model is allowed to see. Creator id is dropped too, so a
        model can't look up the account."""
        return {
            "platform": self.platform,
            "posted_at": self.posted_at,
            "media": [asdict(m) for m in self.media],
            "title": self.title,
            "caption": self.caption,
            "hashtags": list(self.hashtags),
            "duration_s": self.duration_s,
            "creator_niche": self.creator_niche,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Post:
        d = dict(d)
        d["media"] = [Media(**m) for m in d.get("media", [])]
        return cls(**d)
