# Data: YouTube Shorts from small creators

Source: the official **YouTube Data API v3**, using an API key. Nothing is scraped. The only other downloads are thumbnail images, from URLs the API itself returns.

## Setup (one time)

1. In Google Cloud console, create a project and enable **YouTube Data API v3**.
2. Create an **API key**. Restrict it to the YouTube Data API.
3. `export YOUTUBE_API_KEY=...` in your shell. The key is never written to disk by these scripts.

The free quota is 10,000 units/day. One collection run uses about 3,000–5,000 units. Most of that is `search.list`, at 100 units per call.

## Pipeline

```bash
# 1. Collect: discover channels, keep 1k–100k subscribers, list their Shorts
uv run python scripts/collect_youtube.py
#    → data/raw/youtube/<date>/{posts_all.jsonl, channels.jsonl, meta.json, api_cache/}

# 2. Label, split by creator, sample eval sets
uv run python scripts/build_eval_set.py --in data/raw/youtube/<date>/posts_all.jsonl \
  --out-dir data/processed/youtube_<date>

# 3. Thumbnails for the sampled eval sets (~30–100 KB each)
uv run python scripts/fetch_thumbnails.py --thumbs-dir data/raw/youtube/<date>/thumbs \
  data/processed/youtube_<date>/{train,dev,test}.jsonl

# 4. Before each eval session: delete snapshots older than 30 days
uv run python scripts/purge_youtube.py
```

## Collection defaults (`collect_youtube.py --help`)

| Setting | Default | Why |
|---|---|---|
| Published window | 2026-07-01 → 14 days ago | After current models' training cutoffs; views mostly settled after 2 weeks |
| Creator size | 1k–100k subscribers | "Small creators" |
| Shorts per creator | ≥ 25 | Enough to rank a post against the creator's own posts |
| What counts as a Short | Shorts-only playlist (`UUSH…`); fallback: ≤ 180 s and vertical | The API has no official Shorts flag |
| Sponsored posts | dropped at build time | Paid promotion distorts views |

## What a model sees vs. what's hidden

- **Public** (`Post.public_view()`): title, description, hashtags, duration, category, publish time, thumbnail image, and the Short's URL.
  Gemini can watch a YouTube URL natively. Other models get the thumbnail plus text, or frames if we add them later.
- **Hidden**: views, likes, comments, subscriber count, and the label. `assert_no_leak` checks every prompt for them.

## Label

`hidden.label_pctile` (0–100) ranks the Short's `log(1 + views)` against the same channel's ±15 Shorts nearest in time. Use `--metric interactions` to rank on likes + 2×comments instead.

## Policy notes (YouTube API Services Developer Policies)

- Statistics fetched without the channel owner's authorization may be stored for at most **30 days**. Each snapshot is dated, `meta.json` records a `delete_or_refresh_by` date, and `purge_youtube.py` enforces it. To keep an eval set longer, re-collect.
- Don't redistribute the collected data.
- The policy restricts creating "derived metrics" from API data. Our percentile label is an internal eval target, not something shown to users as a YouTube metric. If this becomes a commercial product, review this clause, or use creators' own authorized YouTube Analytics data.
