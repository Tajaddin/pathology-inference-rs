"""Evaluation metrics, numpy only: AUC with bootstrap CI, operating points, calibration, temperature scaling."""

from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int64)
    # Average ranks with ties, vectorised: rank of a value = cumulative count minus half its own ties.
    _, inverse, counts = np.unique(np.asarray(s), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    pos = int(y.sum())
    neg = len(y) - pos
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, metric: Any = auc, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            stats.append(metric(y[idx], s[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_for_specificity(y: np.ndarray, p: np.ndarray, specificity: float) -> float:
    neg = np.sort(p[y == 0])
    k = min(int(np.ceil(specificity * len(neg))), len(neg) - 1)
    return float(neg[k])


def operating_point(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp = float(np.sum(pred & (y == 1)))
    tn = float(np.sum(~pred & (y == 0)))
    return {
        "threshold": float(t),
        "accuracy": float((pred == (y == 1)).mean()),
        "sensitivity": tp / max(float((y == 1).sum()), 1.0),
        "specificity": tn / max(float((y == 0).sum()), 1.0),
        "precision": tp / max(float(pred.sum()), 1.0),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(
        sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any())
    )


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising validation NLL of sigmoid(logit / T); golden-section search on log T."""
    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2

    def f(log_t: float) -> float:
        return nll(y, sigmoid(logits / np.exp(log_t)))

    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))
