"""Compare harnesses on the posts they share, with creator-level bootstrap CIs.

Usage:
    uv run python scripts/compare.py runs/dev300_default:gemini-3.8-flash \
        runs/dev300_rubric:"gemini-3.8-flash|rubric-ranked" ...
The first entry is the baseline for paired differences.
Add --learn-weights RUN:MODEL to also fit category weights for a rubric run
with leave-creators-out cross-validation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.metrics import bootstrap_ci, flop_auroc, pairwise_within_creator, spearman  # noqa: E402

METRICS = {
    "spearman": lambda p, l, c: spearman(p, l),
    "pairwise": pairwise_within_creator,
    "flopAUC": lambda p, l, c: flop_auroc(p, l),
}


def load(spec: str) -> tuple[str, dict[str, dict]]:
    run, model = spec.split(":", 1)
    rows = {}
    for line in open(Path(run) / "scores.jsonl"):
        r = json.loads(line)
        if r.get("model") == model and "score" in r:
            rows[r["post_id"]] = r
    return f"{Path(run).name}:{model}", rows


def paired_diff(metric, a, b, lab, cre, n=2000, seed=0):
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(cre):
        groups.setdefault(c, []).append(i)
    keys = list(groups)
    diffs = []
    for _ in range(n):
        idx, cc = [], []
        for k, c in enumerate(rng.choices(keys, k=len(keys))):
            idx += groups[c]
            cc += [f"{c}#{k}"] * len(groups[c])
        d = (metric([a[i] for i in idx], [lab[i] for i in idx], cc)
             - metric([b[i] for i in idx], [lab[i] for i in idx], cc))
        if d == d:
            diffs.append(d)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1], sum(d > 0 for d in diffs) / len(diffs)


def ridge(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """Solve (X'X + lam I) w = X'y by Gaussian elimination (small d)."""
    d = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j else 0) for j in range(d)] for i in range(d)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(d)]
    for c in range(d):
        p = max(range(c, d), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(c + 1, d):
            f = A[r][c] / A[c][c]
            A[r] = [a - f * ac for a, ac in zip(A[r], A[c])]
            b[r] -= f * b[c]
    w = [0.0] * d
    for c in reversed(range(d)):
        w[c] = (b[c] - sum(A[c][k] * w[k] for k in range(c + 1, d))) / A[c][c]
    return w


def learn_weights(spec: str, folds: int = 5, lam: float = 10.0) -> None:
    from harness.rubric import KEYS, WEIGHTS

    name, rows = load(spec)
    rs = list(rows.values())
    # Within-creator z-scores per category (the "ranked" features), label centred.
    by: dict[str, list[dict]] = {}
    for r in rs:
        by.setdefault(r["creator_id"], []).append(r)
    X, y, cre = [], [], []
    for c, group in by.items():
        for r in group:
            feats = []
            for k in KEYS:
                vals = [g[k] for g in group]
                m = sum(vals) / len(vals)
                sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                feats.append((r[k] - m) / sd if sd else 0.0)
            X.append(feats)
            y.append(r["label_pctile"] - 50)
            cre.append(c)
    creators = sorted(by)
    random.Random(0).shuffle(creators)
    fold_of = {c: i % folds for i, c in enumerate(creators)}
    pred = [0.0] * len(y)
    for f in range(folds):
        tr = [i for i in range(len(y)) if fold_of[cre[i]] != f]
        w = ridge([X[i] for i in tr], [y[i] for i in tr], lam)
        for i in range(len(y)):
            if fold_of[cre[i]] == f:
                pred[i] = sum(wi * xi for wi, xi in zip(w, X[i]))
    lab = [v + 50 for v in y]
    prior = [sum(WEIGHTS[k] * x for k, x in zip(KEYS, xs)) for xs in X]
    print(f"\nLearned category weights for {name} ({folds}-fold, leave-creators-out, ridge λ={lam})")
    for label, p in [("research weights", prior), ("learned weights (CV)", pred)]:
        print(f"  {label:22} " + "  ".join(f"{m} {fn(p, lab, cre):+.2f}" for m, fn in METRICS.items()))
    w = ridge(X, y, lam)
    print("  weights fit on all data (per 1 sd within creator → label points):")
    for k, wi in sorted(zip(KEYS, w), key=lambda t: -t[1]):
        print(f"    {k:16} {wi:+6.2f}   (research weight {WEIGHTS[k]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--learn-weights")
    a = ap.parse_args()

    loaded = [load(s) for s in a.specs]
    ids = sorted(set.intersection(*[set(d) for _, d in loaded]))
    base_name, base = loaded[0]
    lab = [base[i]["label_pctile"] for i in ids]
    cre = [base[i]["creator_id"] for i in ids]
    print(f"{len(ids)} shared posts, {len(set(cre))} creators, {sum(l < 25 for l in lab)} flops (label<25)\n")
    for name, d in loaded:
        p = [d[i]["score"] for i in ids]
        cells = []
        for m, fn in METRICS.items():
            lo, hi = bootstrap_ci(fn, p, lab, cre)
            cells.append(f"{m} {fn(p, lab, cre):+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{name:45} " + "  ".join(cells))
    b = [base[i]["score"] for i in ids]
    for name, d in loaded[1:]:
        p = [d[i]["score"] for i in ids]
        print(f"\n{name} minus {base_name}:")
        for m, fn in METRICS.items():
            lo, hi, pb = paired_diff(fn, p, b, lab, cre)
            print(f"  {m:9} {fn(p, lab, cre) - fn(b, lab, cre):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  P(better)={pb:.2f}")
    if a.learn_weights:
        learn_weights(a.learn_weights)


if __name__ == "__main__":
    main()
