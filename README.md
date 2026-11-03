# Social Media Content Harness & Eval

Can an AI model tell, **before a post goes up**, whether it will perform well? This project builds:

1. **An eval set** of real YouTube Shorts from small creators. Each Short's view count is hidden from the model and used as ground truth.
2. **Scoring harnesses** that rate each Short 0–100: a plain "rate this" prompt as the baseline, and custom harnesses designed to beat it.
3. **Metrics** that measure how well each harness's scores track real performance.

The goal is a tool that helps creators judge a post before publishing, and flags likely flops.

## Results so far

300 Shorts from 30 small creators (1k–100k subscribers), published Jul–Oct 2026, scored by `gemini-3.8-flash`, which watches each Short directly from its YouTube link. 95% confidence intervals come from resampling creators.

| Harness | Rank agreement with real performance (Spearman) | Picks the better of two Shorts from one creator | Spotting bottom-quarter flops (AUROC) |
|---|---|---|---|
| Default prompt ("rate 0–100") | +0.02 [−0.07, +0.11] | 53% | 0.54 |
| **Research rubric, creator-ranked** | **+0.11** [−0.01, +0.22] | 55% | 0.55 |
| Research rubric, raw weighted average | +0.03 | 56% | 0.54 |
| Research rubric, equal weights | +0.11 | 55% | 0.55 |

*Chance level: 0, 50%, 0.50.*

The ranked rubric beats the default prompt on rank agreement by **+0.10**, with a paired-bootstrap probability of 0.95 that it's better. Neither of the other two metrics shows a clear difference.

### What we learned

- **Default prompts rate the channel, not the post.** In the 40-Short pilot, the default prompt's scores varied about 3× more between creators than between one creator's Shorts. The label measures how a Short did *relative to the creator's other Shorts*, so that habit makes the scores useless as predictions.
- **Ranking each Short against the creator's other Shorts is where the gain comes from.** The same rubric ratings, averaged raw, do no better than the default prompt. Converting each category to a within-creator z-score removes the channel-level bias.
- **Picking the better of two Shorts from one creator stays near chance (53–56%).** That's the core skill a pre-post tool needs, and no harness has it yet.
- **No single rubric category is strongly predictive.** At n=300, each category's Spearman correlation falls between −0.03 and +0.07; audio and hook are the strongest. Learned weights did *worse* than fixed weights on held-out creators, so there isn't enough signal yet for weighting to matter.
- **Small pilots mislead.** On 40 Shorts, the rubric's flop AUROC was 0.84. On 300 it was 0.55. The 40-Short sample contained only 4 flops.
- **Views may be a noisy target.** Shorts reach depends heavily on how a seed audience reacts, which involves a lot of luck. Published work that predicts short-video engagement well (SRCC ≈ 0.7, [Li et al., ECCV 2024](https://arxiv.org/abs/2410.00289)) predicts *watch percentage*, which is more stable than view counts.

**Next step:** measure how predictable views are at all, with a noise-ceiling estimate and non-AI baselines such as duration, posting time and channel momentum, before designing further harnesses. See [PLAN.md](PLAN.md) for the full roadmap.

## The harnesses

| Harness | File | Idea | Tokens/Short |
|---|---|---|---|
| `default` | [scripts/run_harness.py](scripts/run_harness.py) | One plain "rate this Short 0–100" call. The baseline to beat. | ~2.4k |
| `anchored` (v1) | [harness/anchored.py](harness/anchored.py) | Also shows 4 of the creator's earlier Shorts with their real view counts, and asks where the new one will land. Fixed the "rates the channel" problem, but accuracy didn't improve on 40 Shorts. | ~12.7k |
| `rubric` (v2) | [harness/rubric.py](harness/rubric.py) | Rates 10 research-based categories 1–10, then combines them **in code**: within-creator z-score per category, weighted sum. | ~3.3k |

### Rubric categories

From [docs/shorts_research.md](docs/shorts_research.md), which summarizes YouTube's own statements on how Shorts are distributed, peer-reviewed engagement research, and creator guidance:

| Category | Weight | Main evidence |
|---|---|---|
| Hook (first 1–3 s) | 3 | YouTube's "viewed vs. swiped away" metric; ad drop-off studies |
| Pacing / retention | 2 | Watch-percentage engagement research |
| Payoff / structure | 2 | Creator guidance (open loops) |
| Emotional pull | 2 | Berger & Milkman 2012: high-arousal emotions spread |
| Originality | 2 | YouTube's Oct 2026 change reducing reach of re-uploads |
| Value / shareability | 1.5 | Berger & Milkman: practical utility |
| Audio | 1 | SnapUGC; large multimodal model results |
| Rewatchability | 1 | Creator guidance (loops) |
| Mobile-native visuals | 0.5 | Production polish barely correlates with engagement (SnapUGC) |
| Packaging (title, hashtags) | 0.5 | Small gain from titles in SnapUGC |

## The eval set

- **Source:** official **YouTube Data API v3**. No scraping. Shorts are identified via each channel's Shorts playlist.
- **Creators:** 1k–100k subscribers, at least 25 Shorts in the window. Sponsored posts are excluded.
- **Window:** published after current models' training cutoffs, and at least 14 days old so view counts have settled.
- **Label:** `label_pctile` (0–100) ranks a Short's `log(views)` against the same creator's ±15 Shorts nearest in time. This removes audience size from the label.
- **Splits:** by creator (train 50% / dev 25% / test 25%), so no creator appears in two splits. The test split is untouched so far.
- **Leakage guard:** views, likes, comments and subscriber counts live in a `hidden` field. Every model request passes `assert_no_leak`, which also blocks the target Short's exact view count.

The first collection had 313 channels and 35,371 Shorts. **The data itself is not in this repo.** YouTube's API policy limits storing statistics to 30 days and forbids redistribution, so you collect your own snapshot (below) and `purge_youtube.py` deletes it after 30 days.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Keys go in .env (git-ignored)
cat > .env <<'EOF'
YOUTUBE_API_KEY=...
GEMINI_API_KEY=...
EOF

uv run pytest                                   # 23 tests, no network

# 1. Collect Shorts (~5k of the 10k daily YouTube quota)
uv run python scripts/collect_youtube.py
# 2. Label, split by creator, sample eval sets
uv run python scripts/build_eval_set.py --in data/raw/youtube/<date>/posts_all.jsonl \
  --out-dir data/processed/youtube_<date>

# 3. Score with a harness
D=data/processed/youtube_<date>
uv run python scripts/run_harness.py --harness default --split-file $D/dev.jsonl \
  --n-creators 30 --per-creator 10 --models gemini-3.8-flash --run-name dev300_default
uv run python scripts/run_harness.py --harness rubric --split-file $D/dev.jsonl \
  --n-creators 30 --per-creator 10 --models gemini-3.8-flash --run-name dev300_rubric

# 4. Compare (first entry is the baseline for paired differences)
uv run python scripts/compare.py runs/dev300_default:gemini-3.8-flash \
  "runs/dev300_rubric:gemini-3.8-flash|rubric-ranked" \
  --learn-weights "runs/dev300_rubric:gemini-3.8-flash|rubric-weighted"
uv run python scripts/report.py runs/dev300_rubric   # includes per-category signal

# Before each session: enforce YouTube's 30-day retention limit
uv run python scripts/purge_youtube.py
```

Model responses are cached under `runs/cache/`, so re-runs only pay for calls not yet made.

## Repo layout

```
evalkit/          schema (public vs hidden fields), leak checks, labels, splits,
                  metrics, YouTube API client, Gemini adapter
harness/          custom harnesses (anchored, rubric)
scripts/          collect → build → run → report/compare, plus purge
docs/             research summary behind the rubric
tests/            unit tests (fake API, synthetic data)
PLAN.md           full project plan and roadmap
data/README.md    data source, policy notes, pipeline details
```

## Limitations

- Only Gemini models have been benchmarked so far; a Claude adapter is planned. Gemini watches the actual video, while other models would see the thumbnail plus text unless frames are added.
- 300 dev Shorts from 30 creators. The confidence intervals are still wide.
- Small-creator Shorts include many compilation and re-upload channels. These aren't filtered yet.
- YouTube's API policy restricts "derived metrics"; the percentile label is used only as an internal eval target. Review this before any commercial use.
