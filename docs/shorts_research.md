# What makes a good YouTube Short: research summary and rubric

Researched 2026-11-03. This summary feeds the rubric harness ([harness/rubric.py](../harness/rubric.py)).

## How Shorts get distributed (why these categories matter)

- **Seed-audience test.** YouTube shows a new Short to a small group of likely-interested viewers. It widens or narrows reach depending on how they respond. YouTube's Shorts lead described the loop as: "Find a set of people that may enjoy your video and depending on how that goes, it may get a lot more traffic" ([Todd Sherman, Creator Insider 2023, via OutlierKit](https://outlierkit.com/resources/youtube-shorts-algorithm/)). Our label is creator-relative view percentile, which is largely a measure of how far each Short got through this loop.
- **"Viewed vs. swiped away"** is the Shorts analytics metric YouTube surfaces: the share of feed impressions where the viewer stayed instead of swiping ([YouTube Help, via OutlierKit](https://outlierkit.com/resources/youtube-shorts-algorithm/)). **Engaged views** count "clicking to watch or watching past the initial seconds" (YouTube blog, 2026, same source). Both put the opening seconds at the center.
- **Originality.** The only confirmed ranking change: on 2026-10-01 YouTube moved "to further prioritize original content and reduce the reach of content re-uploaded from other creators" (YouTube Help community, via [OutlierKit](https://outlierkit.com/resources/youtube-shorts-algorithm/)). Our collection includes many compilation and re-upload channels, so this matters.
- YouTube does **not** publish ranking signals or weights. Specific thresholds in blogs, such as "60% viewed" or "30% swipe-away", are unverified ([OutlierKit](https://outlierkit.com/resources/youtube-shorts-algorithm/), [Gyre](https://gyre.pro/blog/understanding-the-youtube-shorts-algorithm-a--guide)).

## Evidence from research

| Finding | Source | Strength |
|---|---|---|
| Technical video-quality scores correlate only weakly with short-video engagement (SRCC ≈ 0.07–0.31). Aesthetic features add little (0.689→0.696). Visual-emotion features didn't help. | Li et al., *Delving Deep into Engagement Prediction of Short Videos*, ECCV 2024 ([arXiv 2410.00289](https://arxiv.org/abs/2410.00289)): 90k Snapchat Spotlight videos | Strong (peer-reviewed, large) |
| Content and semantic visual features gave the largest gain (0.657→0.689). Background-sound classes helped (0.625→0.636). Titles and descriptions helped (0.636→0.651). | same | Strong |
| A model that hears audio consistently beat one that only sees frames, which suggests audio matters for engagement. | [Engagement Prediction of Short Videos with LMMs, arXiv 2508.02516](https://arxiv.org/abs/2508.02516) (VQualA 2025 challenge winner) | Moderate (model comparison, not an ablation) |
| High-arousal emotions (awe, amusement, anger, anxiety) drive sharing. Low-arousal ones (sadness) reduce it. Practical usefulness also predicts sharing. | Berger & Milkman, *What Makes Online Content Viral?*, JMR 2012 ([summary](https://www.scientificamerican.com/article/the-secret-to-online-success-what-makes-content-go-viral)) | Strong, but on news articles, not Shorts |
| Emotional appeals, personal stories, and background music predicted higher TikTok engagement. Pure text-on-screen formats predicted lower. | [PMC content analysis of TikTok videos](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075627/) | Moderate (one topic) |
| All 12 tested TikTok ads lost >80% of viewers within the first 25% of the video. | [Salminen et al., ACM HT 2024](https://dl.acm.org/doi/pdf/10.1145/3648188.3677048) | Moderate (small n) |
| Across 176k TikToks, median views barely varied by length (58k–76k). Hook and pacing matter more than duration. | [Revid study](https://www.revid.ai/blog/what-makes-a-video-go-viral) | Weak–moderate (vendor) |
| Hook within ~1–3 s; the first frame should work on mute; the subject should sit in the center safe zone; loops, where the ending flows into the start, drive rewatches; open loops that pay off sustain retention. | Creator and industry guides: [JoinBrands](https://joinbrands.com/blog/youtube-shorts-best-practices/), [ShortGenius](https://shortgenius.com/blog/youtube-shorts-best-practices), [Virvid on loops](https://virvid.ai/blog/looping-structure-shorts-retention-2026), [Socialync on open loops](https://www.socialync.io/blog/open-loops-in-content-2026) | Weak (consistent advice, unverified numbers) |

## Rubric categories

Weights reflect strength of evidence and how directly each category ties to the seed-audience loop: swipe-away, watch-through, rewatch, share. They are a-priori. The harness also reports each category's own predictive power, so weights can be learned later.

| # | Category | What it measures | Evidence | Weight |
|---|---|---|---|---|
| 1 | **Hook** | Do the first 1–3 s stop the scroll: visual intrigue, a clear promise or question, works without sound? | Official (swiped-away metric, engaged views); ACM ad drop-off | 3 |
| 2 | **Pacing and retention** | No dead air; steady new information or visual change; trimmed to the essentials | Watch-percentage engagement research; industry | 2 |
| 3 | **Payoff and structure** | Does it set up something (curiosity, tension, a question) and deliver a satisfying payoff or punchline? | Industry (open loops) | 2 |
| 4 | **Emotional pull** | Strength of high-arousal emotion: amusement, awe, surprise, excitement, outrage | Berger & Milkman; PMC study | 2 |
| 5 | **Originality** | Original creation vs. re-upload or compilation of others' content; fresh angle vs. generic template | YouTube 2026-10 originality update | 2 |
| 6 | **Value and shareability** | Useful, relatable, or "send this to a friend": a reason to like, comment, or share | Berger & Milkman (practical utility) | 1.5 |
| 7 | **Audio** | Sound design, music choice, and voice that drive the video, not just background noise | SnapUGC sound gain; LMM audio finding | 1 |
| 8 | **Rewatchability** | Loops seamlessly, or is dense or satisfying enough to replay | Industry (loops; retention >100%) | 1 |
| 9 | **Mobile-native visuals** | Vertical framing, subject centered in the safe zone, legible on-screen text, clear at phone size. *Not* production polish. | Industry; SnapUGC shows polish itself barely matters | 0.5 |
| 10 | **Packaging** | Title, description, and hashtags: clear, intriguing, and matching the content | SnapUGC title/description gain (small) | 0.5 |

### Deliberately excluded or down-weighted
- **Production polish and technical quality**: weak correlation with engagement (SnapUGC). It's folded into category 9 only as phone-legibility.
- **Duration**: didn't help prediction in SnapUGC or the Revid data. Shown to the model as context only.
- **Posting time and frequency**: not content quality. Out of scope for a pre-post content score.

## How scores are combined

1. The model rates each category 1–10 with anchored definitions, as an absolute judgement of the post.
2. **Creator-relative ranking.** Within each creator, each category score is converted to a z-score: how this Short compares with the same creator's other Shorts on that category. This matches the label, which is also creator-relative, and removes the pilot's "rates the channel, not the post" problem.
3. **Final score** = weighted sum of the category z-scores, using the weights above, mapped to 0–100.

Two variants are reported for comparison: (a) the weighted average of raw category scores, without creator ranking, and (b) equal weights.
