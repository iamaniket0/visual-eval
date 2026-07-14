"""Tests for core Soft-TIFA scoring math."""

import math
import random

import pytest

from src.core.scoring import (
    extract_yes_probability,
    probabilities_from_answers,
    soft_tifa_am,
    soft_tifa_gm,
)


def test_am_basic():
    assert soft_tifa_am([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert soft_tifa_am([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert soft_tifa_am([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert soft_tifa_am([]) == pytest.approx(0.0)


def test_gm_basic():
    assert soft_tifa_gm([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    gm = soft_tifa_gm([1.0, 1.0, 0.0], logprob_floor=-10.0)
    assert gm == pytest.approx(math.exp(-10.0 / 3), rel=1e-6)


def test_gm_empty():
    assert soft_tifa_gm([]) == 0.0


def test_gm_leq_am_invariant_random():
    """GM(p_i) <= AM(p_i) for any probabilities in [0, 1]."""
    rng = random.Random(42)
    for _ in range(500):
        n = rng.randint(1, 10)
        probs = [rng.random() for _ in range(n)]
        am = soft_tifa_am(probs)
        gm = soft_tifa_gm(probs, logprob_floor=-10.0)
        assert gm <= am + 1e-9, f"GM {gm} > AM {am} on probs {probs}"


def test_gm_collapses_on_single_confident_miss():
    probs = [0.99, 0.99, 0.99, 0.99, 0.01]
    am = soft_tifa_am(probs)
    gm = soft_tifa_gm(probs, logprob_floor=-10.0)
    assert am > 0.75
    assert gm < 0.50


def test_probabilities_from_soft_answers():
    answers = [
        {"q_id": "q1", "answer": "yes", "probability": 0.95},
        {"q_id": "q2", "answer": "no", "probability": 0.1},
    ]
    probs = probabilities_from_answers(answers)
    assert probs == [0.95, 0.1]


def test_probabilities_from_hard_answers():
    answers = [
        {"q_id": "q1", "answer": "yes"},
        {"q_id": "q2", "answer": "no"},
    ]
    probs = probabilities_from_answers(answers)
    assert probs[0] == 1.0
    assert probs[1] > 0.0
    assert probs[1] < 0.001


def test_probabilities_from_mixed_answers():
    answers = [
        {"answer": "yes"},
        {"answer": "no"},
        {"answer": "yes", "probability": 0.72},
    ]
    probs = probabilities_from_answers(answers, logprob_floor=-10.0)
    assert probs[0] == pytest.approx(1.0)
    assert probs[1] == pytest.approx(math.exp(-10.0))
    assert probs[2] == pytest.approx(0.72)


class _FakeLogprob:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


def test_extract_yes_probability_found():
    top = [
        {"token": "No", "logprob": -2.0},
        {"token": "Yes", "logprob": -0.1},
        {"token": " maybe", "logprob": -5.0},
    ]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.1)) < 1e-6


def test_extract_yes_probability_sdk_shape():
    top = [
        _FakeLogprob("Yes", -0.05),
        _FakeLogprob(" Yes", -2.5),
        _FakeLogprob("No", -4.0),
    ]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert p == pytest.approx(math.exp(-0.05))


def test_extract_yes_probability_not_found():
    top = [
        {"token": "No", "logprob": -0.1},
        {"token": "Nope", "logprob": -2.0},
    ]
    p = extract_yes_probability(top, logprob_floor=-10.0)
    assert abs(p - math.exp(-10.0)) < 1e-10


def test_extract_yes_probability_empty():
    p = extract_yes_probability([], logprob_floor=-10.0)
    assert abs(p - math.exp(-10.0)) < 1e-10
    p2 = extract_yes_probability(None, logprob_floor=-10.0)
    assert abs(p2 - math.exp(-10.0)) < 1e-10
