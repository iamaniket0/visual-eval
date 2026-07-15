"""Tests for bootstrap confidence interval computation."""

from __future__ import annotations

import numpy as np

from src.core.bootstrap import _gm_stat, bootstrap_ci


def test_basic_ci_bounds():
    values = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.88, 0.72]
    lo, hi = bootstrap_ci(values)
    assert lo < hi
    mean_val = np.mean(values)
    assert lo < mean_val < hi


def test_empty_input():
    lo, hi = bootstrap_ci([])
    assert lo == 0.0
    assert hi == 0.0


def test_single_value():
    lo, hi = bootstrap_ci([0.5])
    assert lo == 0.5
    assert hi == 0.5


def test_deterministic():
    values = [0.6, 0.7, 0.8, 0.9, 0.5]
    r1 = bootstrap_ci(values, seed=42)
    r2 = bootstrap_ci(values, seed=42)
    assert r1 == r2


def test_gm_stat():
    values = np.array([0.8, 0.9, 0.7])
    gm = _gm_stat(values)
    assert 0.0 < gm <= float(np.mean(values))


def test_gm_ci():
    values = [0.8, 0.9, 0.7, 0.85, 0.75]
    lo, hi = bootstrap_ci(values, stat_fn=_gm_stat)
    assert lo < hi
    assert 0.0 < lo
    assert hi <= 1.0
