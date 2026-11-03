"""Delete YouTube snapshots older than 30 days, plus eval sets built from them.

YouTube API policy limits storing statistics fetched without the channel
owner's authorization to 30 days. Run this before each eval session (or on a
schedule). Use --dry-run to see what would be deleted.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/youtube"))
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--max-age-days", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=a.max_age_days)).date()
    targets = []
    for d in a.raw_root.glob("*"):
        try:
            if d.is_dir() and datetime.strptime(d.name, "%Y-%m-%d").date() < cutoff:
                targets += [d, a.processed_root / f"youtube_{d.name}"]
        except ValueError:
            continue
    for t in targets:
        if t.exists():
            print(("would delete " if a.dry_run else "deleting ") + str(t))
            if not a.dry_run:
                shutil.rmtree(t)
    if not targets:
        print("nothing older than", cutoff)


if __name__ == "__main__":
    main()
