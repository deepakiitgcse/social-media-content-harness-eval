# Social Media Content Quality: Eval + Custom Harness — Plan

_Current focus: YouTube Shorts from small creators (see Phase 0)._

## Goal

1. **Eval set**: social media posts (image/video + title/caption + metadata) with hidden engagement outcomes (views, likes, follower count).
2. **Benchmark**: ask frontier models (Claude, Gemini, GPT, …) to score each post 0–100 *without* seeing engagement, then measure how closely their scores track real performance.
3. **Custom harness**: a scoring system tuned for this task that beats each model's default "just ask it" setup on a locked test set.
4. **Product use**: give creators a pre-post score, plus a "don't post this" warning for likely underperformers.

---

## Key design decision: what counts as "ground truth"

Raw views and likes mostly measure **audience size**, not content quality. A mediocre post from a creator with 5M followers beats a great post from one with 2K. If the label is raw views, the eval rewards guessing follower count, which isn't the goal.

**Recommended label: creator-relative performance percentile (0–100).**

- For each post, compute an engagement metric, e.g. `log(views)`, or `likes / followers` if views are missing.
- Rank the post against **the same creator's other posts** from a similar time window → percentile 0–100.
- So a score of 80 means "this post did better than 80% of this creator's typical posts."

This removes audience size from the label and keeps the part a creator controls: the content. It also matches the product question, "is *this* post good enough for *me* to publish?"

Secondary labels, reported alongside:
- **Global residual**: residual of `log(engagement)` after regressing on `log(followers)`, platform, post age, and posting time, mapped to a percentile.
- **Raw log-views percentile**: kept only to show how misleading it is.

The models **never** see views, likes, comments, or shares. Follower count is withheld by default. We can run a "with follower count" variant as an ablation.

---

## Phase 0: Scope (decided)

| Decision | Choice |
|---|---|
| Platform | **YouTube Shorts from small creators** (1k–100k subscribers) |
| Data source | Official YouTube Data API v3. No scraping. See `data/README.md` |
| Post content | Title, description, hashtags, duration, category, and thumbnail. Gemini can also watch the Short by URL |
| Window | Published 2026-07-01 onward (after current models' cutoffs) and at least 14 days old |
| Eval size | ~100 creators × 10 Shorts per split (dev/test); full train split kept for retrieval |
| Data retention | Stats stored at most 30 days (API policy); re-collect for later runs |

---

## Phase 1: Build the eval dataset (implemented)

- `scripts/collect_youtube.py` discovers channels through niche searches. It keeps small creators and lists their Shorts through the Shorts-only playlist. Then it fetches stats and keeps channels with at least 25 Shorts in the window.
- `scripts/build_eval_set.py` gives each Short a creator-relative percentile of `log(views)`. It splits creators by stable hash (50/25/25), drops sponsored posts, and samples the eval sets.
- `scripts/fetch_thumbnails.py` downloads thumbnails, using the URLs the API returns.
- `scripts/purge_youtube.py` enforces the 30-day retention limit.
- `evalkit/` holds the schema (public vs. hidden fields), the leak checks, labels, splits, and the API client.

Known limitation: for Shorts, viewers swipe straight into the video, so the thumbnail is a weak proxy for the content. Gemini can use the video URL directly. For other models we'll need frames or a transcript (Phase 4), and we'll report results separately for "text + thumbnail" and "full video".

---

## Phase 2: Eval framework

### 2.1 Repo layout
```
data/            posts.jsonl, media/, splits/
evalkit/
  loader.py      # hides _hidden fields from model inputs
  models/        # adapters: anthropic, gemini, openai (common interface)
  runners/       # default harness, custom harness
  metrics.py
  report.py      # tables + plots, bootstrap CIs
harness/         # custom harness (Phase 3)
runs/            # cached raw outputs per (model, harness, post) — never re-pay for a call
```

### 2.2 Model adapter interface
`score(post) -> {score: float 0-100, rationale: str, raw: ...}`, with retries, response caching keyed by a hash of the prompt and media, and cost and latency logging.

Models: Claude (Opus 5.5, Sonnet 5.5, Haiku 5.5), Gemini (Pro / Flash), GPT. Use temperature 0 where the API supports it.

### 2.3 The "default harness" (baseline to beat)
One plain call per post: system prompt "You are a social media expert", then the post media and text, then "Rate the quality of this post from 0 to 100. Output JSON `{score, reason}`." No examples, rubric, or calibration. This is what someone would get by just asking the model.

### 2.4 Non-LLM baselines (sanity floor)
- Constant 50, and per-niche mean.
- Caption length, hashtag count, posting hour → gradient boosting.
- CLIP or SigLIP image embedding plus a text embedding → ridge regression trained on `train`.
  This one is important. If an embedding regressor beats every LLM, that's a key finding, and its output can feed the custom harness.

### 2.5 Metrics
LLM scores aren't calibrated, so ranking quality matters more than exact numbers.

| Metric | What it measures |
|---|---|
| **Spearman ρ** (primary) | Does the model rank posts in the same order as real performance? |
| **Within-creator pairwise accuracy** | For two posts by the same creator, how often does the model pick the better one? Best match to the product. |
| **AUROC for "flop" detection** | Can the model flag bottom-quartile posts? Directly supports "stop them from posting." |
| MAE vs label percentile | Absolute closeness, after calibration only. |
| Calibration (ECE) | Does a score of 70 mean roughly the 70th percentile? |
| Cost / latency per post | Practical tradeoff. |

All metrics get 95% bootstrap confidence intervals, resampled **by creator**. Harnesses are compared with paired bootstrap on the same posts.

---

## Phase 3: Custom harness

Aim: beat the best default harness on the **locked test set** for Spearman ρ, pairwise accuracy, and flop-AUROC, with non-overlapping CIs. Build it up in layers. Each layer stays only if it improves dev metrics (ablation table).

### Layer 1: Structured rubric extraction (perception)
Have the model output **features** instead of a single number:
- Hook strength in the first frame or first line, visual quality, composition, text legibility
- Clarity of the main subject or message, emotional intensity, novelty or surprise
- Caption quality, call to action, niche and platform fit, trend alignment
- Flags: low light, blurry, cluttered, too much text, off-brand

Each item is a 1–10 rating or a category with short evidence. The model is better at describing content than at predicting engagement, so it handles description and later layers handle prediction.

### Layer 2: Learned aggregator (calibration)
Train a small model (logistic or ridge regression, or LightGBM) on `train`. Inputs: rubric features, image and text embeddings, and simple metadata. Output: predicted label percentile on 0–100.
This turns the model's qualitative judgments into scores grounded in real outcomes. It will likely be the largest single gain.

### Layer 3: Retrieval-augmented few-shot
For each new post, retrieve the k most similar `train` posts by embedding, ideally from the same niche, along with their **actual** percentiles. Show them as anchors, e.g. "Here are 6 similar posts and how they performed." This grounds the model's sense of scale.

### Layer 4: Pairwise and tournament judging
LLMs judge relative quality better than absolute quality. Compare the post against reference posts with known percentiles, e.g. "Which will perform better, A or B?" Then fit a Bradley–Terry / Elo position to get the score. Swap A/B order to cancel position bias.

### Layer 5: Ensembling and self-consistency
- Sample N=3–5 rubric extractions and average them.
- Ensemble across models, e.g. Gemini for video perception and Claude for caption and narrative judgment. The aggregator learns the weights.

### Layer 6: Prompt and program optimization
Use DSPy, or a GEPA- or MIPRO-style optimizer, on `dev` to tune rubric wording, few-shot selection, and instructions, with Spearman and pairwise accuracy as the objective. Guard against overfitting to `dev` by doing a final check only on `test`.

### Layer 7 (optional): Fine-tuning
If enough data is available (>20k posts), fine-tune a smaller model or a vision-language reward head on pairwise preferences. Compare its cost and quality against the prompted harness.

### Output (what the creator sees)
```json
{ "score": 38, "verdict": "revise_before_posting",
  "confidence": "medium",
  "top_issues": ["Weak hook: subject not visible in first frame", "Caption buries the CTA"],
  "suggested_fixes": ["Crop tighter on product", "Lead caption with the question"],
  "similar_posts_performance": "Posts like this landed in the 25th–40th pct for creators in your niche" }
```
The flop threshold is set from the dev ROC curve, picking a target precision so creators aren't flooded with false alarms.

---

## Phase 4: Video support
- **Gemini**: native video input.
- **Claude / GPT**: sample frames (first frame, first 3 s at 2 fps, then evenly spaced) plus an audio transcript (Whisper) plus on-screen text (OCR).
- Report both "native" and "same preprocessed input for all models." Otherwise model comparisons mix up model quality with input-pipeline quality.
- Video-specific rubric items: hook in the first 1–3 s, pacing and cut rate, audio and trending sound, captions on screen, loopability.

---

## Phase 5: Reporting and iteration
- Leaderboard: model × harness × metric, with CIs, cost per post, and latency.
- Ablation table for each custom-harness layer.
- Error analysis: where models are confidently wrong, broken down by niche and media type.
- Label-noise ceiling: estimate how predictable the label is at all, e.g. by checking if near-duplicate posts by the same creator get similar percentiles. Engagement is noisy, and a realistic ρ might be 0.3–0.5. Knowing the ceiling keeps expectations honest.

---

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Data pipeline | 1.5k+ cleaned posts with labels and creator-level splits; leakage test passes |
| M2 | Eval framework and baselines | Default harness run for 3+ models plus embedding baseline; leaderboard generated |
| M3 | Custom harness v1 | Layers 1–3 implemented; beats best default on `dev` |
| M4 | Custom harness v2 | Pairwise, ensembling, and prompt optimization; ablations done |
| M5 | Locked test evaluation | Significant win on `test` for primary metrics; report written |
| M6 | Video and creator-facing output | Video pipeline plus actionable feedback JSON / simple UI |

---

## Risks
- **Label noise**: virality is partly luck. Mitigate with creator-relative labels, enough posts per creator, and an honest noise ceiling.
- **Training-data contamination**: mitigate with post-cutoff posts and a separately reported subset.
- **Overfitting the harness to dev**: mitigate with a locked test set, creator-level splits, and few test runs.
- **Cost**: mitigate with response caching, cheap models for perception, and expensive models only where ablations show they help.
- **Data access and ToS**: prefer official APIs or creator-owned data.
