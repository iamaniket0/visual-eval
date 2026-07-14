"""Tests for core utilities."""
from src.core.utils import CostTracker


def test_cost_tracker_hard_cap():
    ct = CostTracker(hard_cap_usd=1.0)
    assert ct.check_cap()
    ct.add(0.5, model="m", stage="generation")
    assert ct.check_cap()
    ct.add(0.6, model="m", stage="generation")
    assert not ct.check_cap()


def test_cost_tracker_summary():
    ct = CostTracker(hard_cap_usd=100.0)
    ct.add(1.5, model="flux", stage="generation")
    ct.add(0.5, model="bria", stage="editing")
    s = ct.summary()
    assert s["total_usd"] == 2.0
    assert s["by_model"]["flux"] == 1.5
    assert s["by_stage"]["editing"] == 0.5
