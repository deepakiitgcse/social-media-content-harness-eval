"""Download thumbnails (URLs returned by the YouTube Data API) for posts in
the given jsonl files, and add them as image media. Files are rewritten in
place. Already-downloaded thumbnails are skipped.

Usage:
    uv run python scripts/fetch_thumbnails.py --thumbs-dir data/raw/youtube/<date>/thumbs \
        data/processed/youtube_<date>/{train,dev,test}.jsonl
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.loader import read_posts, write_posts  # noqa: E402
from evalkit.schema import Media  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--thumbs-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    a = ap.parse_args()
    a.thumbs_dir.mkdir(parents=True, exist_ok=True)
    root = a.data_root.resolve()
    fetched = failed = 0

    for f in a.files:
        posts = list(read_posts(f))
        for p in posts:
            dest = a.thumbs_dir / f"{p.post_id}.jpg"
            url = p.hidden.get("thumbnail_url")
            if not dest.exists() and url:
                try:
                    with urllib.request.urlopen(url, timeout=20) as r:
                        dest.write_bytes(r.read())
                    fetched += 1
                except OSError as e:
                    failed += 1
                    print(f"{p.post_id}: {e}", file=sys.stderr)
                    continue
            rel = str(dest.resolve().relative_to(root))
            if dest.exists() and not any(m.path == rel for m in p.media):
                p.media.append(Media("image", path=rel))
        write_posts(posts, f)
    print(f"fetched={fetched} failed={failed}")


if __name__ == "__main__":
    main()
