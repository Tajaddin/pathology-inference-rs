"""Evaluation metrics against scikit-learn and against known answers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pcam import metrics


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = y * 0.8 + rng.normal(0, 0.6, len(y))
    return y, s


def test_auc_matches_sklearn(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = scored
    assert metrics.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_handles_ties_like_sklearn() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.4, 0.4, 0.9, 0.4, 0.2, 0.9, 0.2])
    assert metrics.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_bootstrap_interval_contains_the_estimate(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = scored
    lo, hi = metrics.bootstrap_ci(y, s, n_boot=200)
    assert lo < metrics.auc(y, s) < hi
    assert hi - lo < 0.05


def test_threshold_reaches_the_requested_specificity(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y, s = scored
    p = metrics.sigmoid(s)
    t = metrics.threshold_for_specificity(y, p, 0.95)
    assert metrics.operating_point(y, p, t)["specificity"] >= 0.95


def test_operating_point_counts() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.7, 0.1])
    op = metrics.operating_point(y, p, 0.5)
    assert (op["sensitivity"], op["specificity"], op["precision"], op["accuracy"]) == (0.5, 0.5, 0.5, 0.5)


def test_ece_is_zero_for_calibrated_and_large_for_overconfident() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, len(p)) < p).astype(int)
    assert metrics.ece(y, p) < 0.01
    assert metrics.ece(y, np.where(p > 0.5, 0.999, 0.001)) > 0.2


def test_temperature_recovers_the_true_one() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(0, 4, 100_000)
    y = (rng.uniform(0, 1, len(z)) < metrics.sigmoid(z / 2.5)).astype(int)
    assert metrics.fit_temperature(z, y) == pytest.approx(2.5, rel=0.03)


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    out = metrics.sigmoid(np.array([-1e4, 0.0, 1e4]))
    assert np.all(np.isfinite(out))
    assert out[1] == 0.5
