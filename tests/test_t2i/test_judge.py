"""Tests for T2I judge (no API calls)."""
import math
import pytest

from src.t2i.judge import (
    JudgeResult, _extract_json, _format_questions,
    GPT4oHardJudge, GPT4oSoftJudge, QwenSoftJudge, TogetherQwen35SoftJudge,
    SoftTifaLogprobsUnavailable,
    judge_client_factory, JudgeClient,
)


def test_extract_json_handles_fences():
    raw = '```json\n{"answers": [{"q_id": "q1", "answer": "yes"}]}\n```'
    assert _extract_json(raw) == {"answers": [{"q_id": "q1", "answer": "yes"}]}


def test_format_questions_output():
    qs = [{"q_id": "q1", "question": "Is there a cat?", "type": "presence"},
          {"q_id": "q2", "question": "Is the cat red?", "type": "attribute"}]
    text = _format_questions(qs)
    assert "q1" in text and "Is there a cat?" in text


def test_judge_client_alias_points_to_hard_judge():
    assert JudgeClient is GPT4oHardJudge


def test_judge_result_scoring():
    r = JudgeResult(
        prompt_id="p1", model="m1", image_path="x.png", judge_model="gpt-4o",
        answers=[{"q_id": "q1", "answer": "yes"},
                 {"q_id": "q2", "answer": "no"},
                 {"q_id": "q3", "answer": "yes"}],
        score=2/3,
    )
    d = r.to_dict()
    assert d["prompt_id"] == "p1"
    assert d["score"] == pytest.approx(0.6667, rel=1e-3)
    assert len(d["answers"]) == 3


def test_factory_picks_qwen_together_soft(monkeypatch):
    import src.t2i.judge as judge_mod
    monkeypatch.setattr(judge_mod, "load_settings", lambda: {
        "judge": {"backend": "qwen_together_soft",
                   "model_slug": "Qwen/Qwen3.5-397B-A17B",
                   "logprob_floor": -10},
    })
    monkeypatch.setenv("TOGETHER_API_KEY", "tgp_v1_test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    client = judge_client_factory()
    assert isinstance(client, TogetherQwen35SoftJudge)
    assert client.api_key == "tgp_v1_test"
    assert "together.xyz" in (client._base_url or "")


def test_factory_picks_gpt4o_hard(monkeypatch):
    import src.t2i.judge as judge_mod
    monkeypatch.setattr(judge_mod, "load_settings", lambda: {
        "api_routing": {"judge": "openrouter"},
        "judge": {"backend": "gpt4o_hard"},
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    client = judge_client_factory()
    assert isinstance(client, GPT4oHardJudge)


def test_factory_override_backend(monkeypatch):
    import src.t2i.judge as judge_mod
    monkeypatch.setattr(judge_mod, "load_settings", lambda: {
        "api_routing": {"judge": "openrouter"},
        "judge": {"backend": "gpt4o_hard"},
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    client = judge_client_factory(override_backend="gpt4o_soft")
    assert isinstance(client, GPT4oSoftJudge)
