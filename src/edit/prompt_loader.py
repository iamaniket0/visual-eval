"""Prompt loader for edit-eval.

Loads the two prompt files:
    prompts/layer1_gold.json       (120 prompts from GEditBench v2 / EditBench)
    prompts/layer2_proprietary.json (45 proprietary prompts)

Prompt schema:
    {
      "prompt_id": "L1_IBF_001",
      "layer": 1,
      "sub_category": "instruction_boundary",
      "difficulty": "medium",
      "source_image": "source_images/L1_IBF_001.jpg",
      "edit_instruction": "Change the person's shirt from white to navy blue",
      "turns": 1,
      "atoms": [
        {"q_id": "q1", "question": "...", "type": "instruction", "dimension": "instruction_following"},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.utils import get_logger
from src.edit import PROMPTS_DIR

log = get_logger("prompt_loader")


def load_layer1_gold(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or (PROMPTS_DIR / "layer1_gold.json")
    if not p.exists():
        log.warning("%s not found", p)
        return []
    with open(p) as f:
        return json.load(f)


def load_layer2_proprietary(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or (PROMPTS_DIR / "layer2_proprietary.json")
    if not p.exists():
        log.warning("%s not found", p)
        return []
    with open(p) as f:
        return json.load(f)


def load_all_prompts() -> list[dict[str, Any]]:
    """Load and merge both prompt layers."""
    prompts = load_layer1_gold() + load_layer2_proprietary()
    log.info(
        "Loaded %d prompts (L1=%d, L2=%d)",
        len(prompts),
        sum(1 for p in prompts if p.get("layer") == 1),
        sum(1 for p in prompts if p.get("layer") == 2),
    )
    return prompts


def prompts_by_id(prompts: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    if prompts is None:
        prompts = load_all_prompts()
    return {p["prompt_id"]: p for p in prompts}


def resolve_source_image_path(prompt: dict[str, Any]) -> str:
    """Resolve the source_image field to an absolute path."""
    src = prompt.get("source_image", "")
    if not src:
        return ""
    p = PROMPTS_DIR / src
    return str(p) if p.exists() else src
