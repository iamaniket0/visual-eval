"""Integration test for the T2I aggregator using synthetic judgments."""

import json

import pandas as pd
import pytest

from src.t2i import aggregator


@pytest.fixture
def fake_outputs(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    prompts_dir = tmp_path / "prompts"
    (outputs / "judgments").mkdir(parents=True)
    (outputs / "metadata").mkdir(parents=True)
    (outputs / "scores").mkdir(parents=True)
    prompts_dir.mkdir(parents=True)

    monkeypatch.setattr(aggregator, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(aggregator, "SCORES_DIR", outputs / "scores")
    monkeypatch.setattr(aggregator, "PROMPTS_DIR", prompts_dir)

    prompts = [
        {
            "prompt_id": "L1_NUM_001",
            "layer": 1,
            "sub_category": "numeracy",
            "difficulty": "auto",
            "prompt_text": "three cats",
            "atomic_questions": [{"q_id": "q1", "question": "cats?", "type": "presence"}],
        },
        {
            "prompt_id": "L2_NUM_001",
            "layer": 2,
            "sub_category": "numeracy",
            "difficulty": "medium",
            "prompt_text": "seven bottles",
            "atomic_questions": [{"q_id": "q1", "question": "bottles?", "type": "presence"}],
        },
    ]
    with open(prompts_dir / "prompt_set.json", "w") as f:
        json.dump(prompts, f)

    from src.t2i import prompt_loader

    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", prompts_dir)

    def write_jsonl(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    write_jsonl(
        outputs / "judgments" / "model_a.jsonl",
        [
            {
                "prompt_id": "L1_NUM_001",
                "model": "model_a",
                "image_path": "x.png",
                "score": 0.9,
                "answers": [
                    {"q_id": "q1", "question": "cats?", "type": "presence", "answer": "yes"}
                ],
            },
            {
                "prompt_id": "L2_NUM_001",
                "model": "model_a",
                "image_path": "x.png",
                "score": 0.3,
                "answers": [
                    {"q_id": "q1", "question": "bottles?", "type": "presence", "answer": "no"}
                ],
            },
        ],
    )
    write_jsonl(
        outputs / "judgments" / "model_b.jsonl",
        [
            {
                "prompt_id": "L1_NUM_001",
                "model": "model_b",
                "image_path": "y.png",
                "score": 0.5,
                "answers": [
                    {"q_id": "q1", "question": "cats?", "type": "presence", "answer": "no"}
                ],
            },
            {
                "prompt_id": "L2_NUM_001",
                "model": "model_b",
                "image_path": "y.png",
                "score": 0.5,
                "answers": [
                    {"q_id": "q1", "question": "bottles?", "type": "presence", "answer": "yes"}
                ],
            },
        ],
    )

    with open(prompts_dir / "prompt_themes.json", "w") as f:
        json.dump(
            {
                "L1_NUM_001": ["animals", "few-objects", "counting"],
                "L2_NUM_001": ["objects", "dense", "material-heavy", "counting"],
            },
            f,
        )

    return outputs


def test_aggregation_produces_expected_files(fake_outputs):
    paths = aggregator.run_aggregation()
    assert "leaderboard" in paths and paths["leaderboard"].exists()
    lb = pd.read_csv(paths["leaderboard"])
    assert set(lb["model"]) == {"model_a", "model_b"}
    assert lb.iloc[0]["model"] == "model_a"


def test_layer_comparison_detects_divergence(fake_outputs):
    paths = aggregator.run_aggregation()
    lc = pd.read_csv(paths["layer_comparison"])
    row_a = lc[lc["model"] == "model_a"].iloc[0]
    assert row_a["layer1_gold"] == pytest.approx(0.9)
    assert row_a["layer2_proprietary"] == pytest.approx(0.3)
    assert row_a["divergence"] == pytest.approx(0.6)
