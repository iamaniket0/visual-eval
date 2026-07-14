"""Tests for T2I prompt loader."""

import json

import pytest

from src.t2i.prompt_loader import (
    _extract_json,
    _placeholder_decomposition,
    _prompt_id,
    stratified_sample,
)


def test_stratified_sample_returns_requested_size():
    prompts = [f"prompt {i}" * i for i in range(1, 31)]
    sample = stratified_sample(prompts, n=9, seed=1)
    assert len(sample) == 9
    assert len(set(sample)) == 9


def test_stratified_sample_handles_small_pool():
    prompts = ["a", "b", "c"]
    sample = stratified_sample(prompts, n=10, seed=1)
    assert sorted(sample) == ["a", "b", "c"]


def test_stratified_sample_is_deterministic():
    prompts = [f"prompt {i}" for i in range(50)]
    a = stratified_sample(prompts, n=10, seed=42)
    b = stratified_sample(prompts, n=10, seed=42)
    assert a == b


def test_prompt_id_format():
    assert _prompt_id(1, "numeracy", 5) == "L1_NUM_005"
    assert _prompt_id(2, "complex_compositions", 17) == "L2_CMP_017"
    assert _prompt_id(1, "spatial_3d", 150) == "L1_SP3_150"


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_code_fence():
    raw = '```json\n{"questions": []}\n```'
    assert _extract_json(raw) == {"questions": []}


def test_extract_json_raises_when_no_json():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _extract_json("no json here")


def test_placeholder_decomposition_shape():
    result = _placeholder_decomposition("A cat on a mat")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all("q_id" in q and "question" in q and "type" in q for q in result)
