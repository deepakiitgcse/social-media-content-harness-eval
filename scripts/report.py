"""Summarise a run: per-model metrics, score spread, cross-model agreement.

Usage: uv run python scripts/report.py runs/pilot
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.metrics import (  # noqa: E402
    bootstrap_ci, flop_auroc, pairwise_within_creator, spearman,
)


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(run / "scores.jsonl")]
    by_model = defaultdict(dict)
    for r in rows:
        if "score" in r:
            by_model[r["model"]][r["post_id"]] = r

    print(f"{'model':24} {'n':>3} {'mean':>5} {'sd':>5} {'min':>4} {'max':>4} {'uniq':>4}  "
          f"{'spearman [95% CI]':>22}  {'pairwise':>8}  {'flopAUC':>7}  {'tok/post':>8}  {'s/post':>6}")
    for m, d in by_model.items():
        rs = list(d.values())
        pred = [r["score"] for r in rs]
        lab = [r["label_pctile"] for r in rs]
        cre = [r["creator_id"] for r in rs]
        rho = spearman(pred, lab)
        lo, hi = bootstrap_ci(lambda p, l, c: spearman(p, l), pred, lab, cre)
        tok = st.mean((r.get("usage") or {}).get("totalTokenCount", 0) for r in rs)
        lat = st.mean(r.get("latency_s") or 0 for r in rs)
        print(f"{m:24} {len(rs):3d} {st.mean(pred):5.1f} {st.pstdev(pred):5.1f} {min(pred):4.0f} "
              f"{max(pred):4.0f} {len(set(pred)):4d}  {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]       "
              f"{pairwise_within_creator(pred, lab, cre):8.2f}  {flop_auroc(pred, lab):7.2f}  "
              f"{tok:8.0f}  {lat:6.1f}")

    models = list(by_model)
    if len(models) > 1:
        print("\nCross-model Spearman (do the models agree with each other?)")
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ids = sorted(set(by_model[a]) & set(by_model[b]))
                print(f"  {a} vs {b}: {spearman([by_model[a][k]['score'] for k in ids], [by_model[b][k]['score'] for k in ids]):+.2f}  (n={len(ids)})")

    rubric_rows = {m: d for m, d in by_model.items() if m.endswith("|rubric-weighted")}
    for m, d in rubric_rows.items():
        from harness.rubric import KEYS, WEIGHTS
        rs = list(d.values())
        lab = [r["label_pctile"] for r in rs]
        cre = [r["creator_id"] for r in rs]
        print(f"\nPer-category signal ({m.split('|')[0]}): does each category alone predict performance?")
        print(f"  {'category':16} {'weight':>6} {'mean':>5} {'spearman':>8} {'pairwise':>8}")
        for k in KEYS:
            v = [r[k] for r in rs]
            print(f"  {k:16} {WEIGHTS[k]:6.1f} {st.mean(v):5.1f} {spearman(v, lab):+8.2f} "
                  f"{pairwise_within_creator(v, lab, cre):8.2f}")

    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} errors, e.g. {errs[0]['model']}: {str(errs[0]['error'])[:200]}")


if __name__ == "__main__":
    main()
