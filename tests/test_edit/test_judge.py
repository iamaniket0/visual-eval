"""Tests for edit-eval judge utility functions."""

import math

from src.core.scoring import extract_yes_probability, soft_tifa_am, soft_tifa_gm


def test_extract_yes_probability_found():
    top_logprobs = [
        {"token": "No", "logprob": -2.0},
        {"token": "Yes", "logprob": -0.1},
        {"token": " maybe", "logprob": -5.0},
    ]
    p = extract_yes_probability(top_logprobs, logprob_floor=-10.0)
    assert abs(p - math.exp(-0.1)) < 1e-6


def test_extract_yes_probability_not_found():
    top_logprobs = [
        {"token": "No", "logprob": -0.1},
        {"token": "Nope", "logprob": -2.0},
    ]
    p = extract_yes_probability(top_logprobs, logprob_floor=-10.0)
    assert abs(p - math.exp(-10.0)) < 1e-10


def test_am_gm_invariant():
    probs = [0.8, 0.6, 0.9, 0.3]
    assert soft_tifa_gm(probs, -10.0) <= soft_tifa_am(probs)


def test_am_single():
    assert soft_tifa_am([0.75]) == 0.75


def test_gm_single():
    assert abs(soft_tifa_gm([0.75], -10.0) - 0.75) < 1e-6
