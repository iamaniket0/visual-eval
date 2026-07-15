"""Bootstrap confidence intervals for prompt-level score aggregation."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from src.core.scoring import DEFAULT_LOGPROB_FLOOR


def _gm_stat(values: np.ndarray) -> float:
    floor_p = math.exp(DEFAULT_LOGPROB_FLOOR)
    clamped = np.maximum(values, floor_p)
    return float(np.exp(np.mean(np.log(clamped))))


def bootstrap_ci(
    values: list[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,  # type: ignore[assignment]
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI bounds via percentile method."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot_stats = np.array(
        [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_stats, 100 * alpha)),
        float(np.percentile(boot_stats, 100 * (1 - alpha))),
    )
