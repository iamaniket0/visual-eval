"""Tests for the edit-eval aggregator scoring math."""

import math

from src.core.scoring import soft_tifa_am, soft_tifa_gm, probabilities_from_answers


def test_am_basic():
    assert soft_tifa_am([1.0, 0.5, 0.0]) == 0.5


def test_am_empty():
    assert soft_tifa_am([]) == 0.0


def test_am_all_ones():
    assert soft_tifa_am([1.0, 1.0, 1.0]) == 1.0


def test_gm_basic():
    probs = [0.9, 0.8, 0.7]
    expected = math.exp(sum(math.log(p) for p in probs) / 3)
    assert abs(soft_tifa_gm(probs) - expected) < 1e-6


def test_gm_with_zero():
    result = soft_tifa_gm([1.0, 0.0, 1.0], logprob_floor=-10.0)
    assert result > 0.0
    assert result < 1.0


def test_gm_empty():
    assert soft_tifa_gm([]) == 0.0


def test_gm_le_am():
    probs = [0.9, 0.3, 0.7, 0.5]
    assert soft_tifa_gm(probs) <= soft_tifa_am(probs)


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
