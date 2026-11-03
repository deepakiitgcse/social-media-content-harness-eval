"""Eval metrics. Pure stdlib; all take parallel lists of predictions and labels."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence


def ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def spearman(pred: Sequence[float], label: Sequence[float]) -> float:
    return pearson(ranks(pred), ranks(label))


def pairwise_within_creator(pred, label, creators) -> float:
    """Of all same-creator pairs with different labels, fraction where the
    model ranks them the same way (prediction ties count as half)."""
    by = defaultdict(list)
    for p, l, c in zip(pred, label, creators):
        by[c].append((p, l))
    agree = total = 0.0
    for items in by.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (pi, li), (pj, lj) = items[i], items[j]
                if li == lj:
                    continue
                total += 1
                if pi == pj:
                    agree += 0.5
                elif (pi > pj) == (li > lj):
                    agree += 1
    return agree / total if total else float("nan")


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Probability a random positive outranks a random negative."""
    pos = [s for s, y in zip(scores, positive) if y]
    neg = [s for s, y in zip(scores, positive) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def flop_auroc(pred, label, threshold: float = 25.0) -> float:
    """How well LOW scores pick out bottom-quartile posts."""
    return auroc([-p for p in pred], [l < threshold for l in label])


def bootstrap_ci(metric: Callable[..., float], pred, label, creators,
                 n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI, resampling whole creators."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, c in enumerate(creators):
        by[c].append(i)
    keys = list(by)
    vals = []
    for _ in range(n):
        idx, cre = [], []
        # A creator drawn twice counts as two separate groups.
        for k, c in enumerate(rng.choices(keys, k=len(keys))):
            idx += by[c]
            cre += [f"{c}#{k}"] * len(by[c])
        v = metric([pred[i] for i in idx], [label[i] for i in idx], cre)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
