"""Mocked tests for judge logprobs parsing and error handling."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.scoring import extract_yes_probability

# ---------------------------------------------------------------------------
# extract_yes_probability unit tests (no HTTP needed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeLogprob:
    token: str
    logprob: float


def test_yes_token_found():
    top = [_FakeLogprob("Yes", -0.2), _FakeLogprob("No", -3.0)]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.2)) < 1e-6


def test_yes_with_leading_space():
    top = [_FakeLogprob(" Yes", -0.1), _FakeLogprob(" No", -4.0)]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.1)) < 1e-6


def test_lowercase_yes():
    top = [_FakeLogprob("yes", -0.3)]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.3)) < 1e-6


def test_no_yes_token_returns_floor():
    top = [_FakeLogprob("No", -0.01), _FakeLogprob("Maybe", -5.0)]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-10.0)) < 1e-6


def test_empty_logprobs_returns_floor():
    p = extract_yes_probability([], logprob_floor=-10.0)
    assert abs(p - math.exp(-10.0)) < 1e-6


def test_dict_format_logprobs():
    top = [{"token": "Yes", "logprob": -0.5}, {"token": "No", "logprob": -1.0}]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.5)) < 1e-6


def test_best_yes_variant_selected():
    top = [
        _FakeLogprob("Yes", -0.5),
        _FakeLogprob(" Yes", -0.1),
        _FakeLogprob("No", -3.0),
    ]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.1)) < 1e-6


# ---------------------------------------------------------------------------
# CostTracker alert callback
# ---------------------------------------------------------------------------


def test_cost_tracker_alert(caplog):
    import logging

    from src.core.utils import CostTracker

    ct = CostTracker(hard_cap_usd=10.0, alert_at_fraction=0.8)
    with caplog.at_level(logging.WARNING, logger="cost"):
        ct.add(7.0, model="m", stage="generation")
        ct.add(1.5, model="m", stage="generation")

    assert any("80%" in rec.message or "85%" in rec.message for rec in caplog.records)


def test_cost_tracker_thread_safety():
    import threading

    from src.core.utils import CostTracker

    ct = CostTracker(hard_cap_usd=1000.0)
    n_threads = 10
    adds_per = 100
    amount = 0.01

    def worker():
        for _ in range(adds_per):
            ct.add(amount, model="m", stage="s")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * adds_per * amount
    assert abs(ct.total - expected) < 1e-6
