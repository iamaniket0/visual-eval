"""Validate prompt difficulty against Complex-Edit C1-C8 scale.

Checks:
1. Word count within difficulty band
2. Atom count within difficulty band
3. Atomic operations count matches difficulty
4. Hard prompts have spatial/physics/multi-object requirements
5. Easy prompts don't have hidden complexity
6. All source images exist
7. All atoms have valid dimensions
8. Multi-turn prompts have turns array

Usage:
    python scripts/validate_prompt_difficulty.py
    python scripts/validate_prompt_difficulty.py --fix   # auto-reclassify mismatches
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "config" / "prompt_taxonomy.yaml"

VALID_DIMENSIONS = {"instruction_following", "visual_consistency", "detail_preservation"}
VALID_ATOM_TYPES = {
    "instruction",
    "preservation",
    "identity",
    "structure",
    "quality",
    "lighting",
    "placement",
}

COMPLEXITY_MARKERS = {
    "spatial": [
        "left",
        "right",
        "behind",
        "in front",
        "above",
        "below",
        "next to",
        "opposite side",
        "viewpoint",
        "perspective",
        "parallax",
        "vanishing point",
    ],
    "physics": [
        "shadow",
        "reflection",
        "refraction",
        "caustic",
        "mirror",
        "chrome",
        "transparent",
        "water flow",
        "motion blur",
        "wet",
    ],
    "multi_object": ["all ", "every ", "each ", "both ", "everyone"],
    "compound": [" and ", " while ", " with correct", " matching the", " consistent with"],
}


def count_atomic_ops(instruction: str) -> int:
    """Estimate number of atomic operations in an instruction."""
    ops = 1
    for marker in [" and ", " → ", ", then ", " while also ", " plus "]:
        ops += instruction.lower().count(marker)
    if ":" in instruction and any(c in instruction for c in ["flat", "strong", "wave"]):
        ops += instruction.count(",") // 2
    return min(ops, 8)


def has_complexity(instruction: str, kind: str) -> bool:
    markers = COMPLEXITY_MARKERS.get(kind, [])
    lower = instruction.lower()
    return any(m in lower for m in markers)


def validate_prompt(prompt: dict, diff_config: dict) -> list[str]:
    """Return list of warning strings for a single prompt."""
    warnings = []
    pid = prompt.get("prompt_id", "???")
    diff = prompt.get("difficulty", "???")
    instr = prompt.get("edit_instruction", "")
    atoms = prompt.get("atoms", [])
    source = prompt.get("source_image", "")

    if diff not in diff_config:
        warnings.append(f"{pid}: unknown difficulty '{diff}'")
        return warnings

    cfg = diff_config[diff]

    # Word count check
    wc = len(instr.split())
    wc_min, wc_max = cfg["word_count"]
    if wc < wc_min - 2:
        warnings.append(f"{pid}: word count {wc} below {diff} minimum {wc_min}")
    if wc > wc_max + 10:
        warnings.append(f"{pid}: word count {wc} above {diff} maximum {wc_max}")

    # Atom count check
    ac = len(atoms)
    ac_min, ac_max = cfg["atom_count"]
    if ac < ac_min:
        warnings.append(f"{pid}: atom count {ac} below {diff} minimum {ac_min}")
    if ac > ac_max + 2:
        warnings.append(f"{pid}: atom count {ac} above {diff} maximum {ac_max}")

    # Complexity check for easy prompts
    if diff == "easy":
        atomic_ops = count_atomic_ops(instr)
        if atomic_ops > 2:
            warnings.append(f"{pid}: EASY but has {atomic_ops} atomic ops (should be 1-2)")
        if has_complexity(instr, "spatial") and "flip" not in instr.lower():
            # Allow simple directional mentions like "shadow under"
            spatial_words = [m for m in COMPLEXITY_MARKERS["spatial"] if m in instr.lower()]
            non_trivial = [
                w for w in spatial_words if w not in ("left", "right", "behind", "above", "below")
            ]
            if non_trivial:
                warnings.append(f"{pid}: EASY but has spatial complexity markers: {non_trivial}")
        if has_complexity(instr, "physics"):
            # Allow simple shadow/reflection additions — they ARE easy (C1)
            physics_words = [m for m in COMPLEXITY_MARKERS["physics"] if m in instr.lower()]
            is_simple_shadow = (
                len(physics_words) == 1 and physics_words[0] == "shadow" and atomic_ops <= 2
            )
            if not is_simple_shadow:
                warnings.append(f"{pid}: EASY but has physics complexity markers: {physics_words}")

    # Hard prompts should have complexity
    if diff == "hard":
        has_any = (
            has_complexity(instr, "spatial")
            or has_complexity(instr, "physics")
            or has_complexity(instr, "multi_object")
            or has_complexity(instr, "compound")
        )
        if not has_any:
            warnings.append(f"{pid}: HARD but no spatial/physics/multi-object/compound markers")

    # Atom dimension validation
    for atom in atoms:
        dim = atom.get("dimension", "")
        if dim not in VALID_DIMENSIONS:
            warnings.append(f"{pid}: atom {atom.get('q_id')} has invalid dimension '{dim}'")
        atype = atom.get("type", "")
        if atype not in VALID_ATOM_TYPES:
            warnings.append(f"{pid}: atom {atom.get('q_id')} has invalid type '{atype}'")

    # Source image exists
    img_path = ROOT / "prompts" / source
    if not img_path.exists():
        warnings.append(f"{pid}: source image missing: {source}")

    # Multi-turn check
    cat = prompt.get("category", prompt.get("sub_category", ""))
    if cat == "multi_turn":
        turns = prompt.get("turns", 1)
        if not isinstance(turns, list):
            warnings.append(f"{pid}: multi_turn but turns is not a list")
        elif len(turns) < 2:
            warnings.append(f"{pid}: multi_turn but only {len(turns)} turn(s)")

    # Instruction atoms check (at least 1 instruction atom)
    instr_atoms = [a for a in atoms if a.get("type") == "instruction"]
    if not instr_atoms:
        warnings.append(f"{pid}: no instruction-type atoms")

    return warnings


def main(args):
    taxonomy = yaml.safe_load(TAXONOMY_PATH.read_text())
    diff_config = taxonomy["difficulty"]

    all_warnings = []
    total = 0
    stats = {"clean": 0, "warnings": 0}

    for fname in ["prompts/layer1_gold.json", "prompts/layer2_proprietary.json"]:
        path = ROOT / fname
        if not path.exists():
            print(f"SKIP: {fname} not found")
            continue

        prompts = json.loads(path.read_text())
        print(f"\n{'=' * 60}")
        print(f"Validating {fname}: {len(prompts)} prompts")
        print(f"{'=' * 60}")

        file_warnings = []
        for p in prompts:
            total += 1
            ws = validate_prompt(p, diff_config)
            if ws:
                file_warnings.extend(ws)
                stats["warnings"] += 1
            else:
                stats["clean"] += 1

        if file_warnings:
            print(f"\n  Warnings ({len(file_warnings)}):")
            for w in file_warnings[:30]:
                print(f"    ⚠ {w}")
            if len(file_warnings) > 30:
                print(f"    ... and {len(file_warnings) - 30} more")
        else:
            print("  ✓ All prompts pass validation")

        all_warnings.extend(file_warnings)

    # Category-difficulty coverage matrix
    print(f"\n{'=' * 60}")
    print("Coverage Matrix (category × difficulty)")
    print(f"{'=' * 60}")

    all_prompts = []
    for fname in ["prompts/layer1_gold.json", "prompts/layer2_proprietary.json"]:
        path = ROOT / fname
        if path.exists():
            all_prompts.extend(json.loads(path.read_text()))

    matrix = {}
    for p in all_prompts:
        cat = p.get("category", p.get("sub_category", "???"))
        diff = p.get("difficulty", "???")
        matrix.setdefault(cat, {}).setdefault(diff, 0)
        matrix[cat][diff] += 1

    min_per_cell = taxonomy["targets"]["min_per_cell"]
    print(f"\n{'Category':<20} {'Easy':>6} {'Medium':>8} {'Hard':>6} {'Total':>7}")
    print("-" * 50)
    gaps = []
    for cat in sorted(matrix.keys()):
        e = matrix[cat].get("easy", 0)
        m = matrix[cat].get("medium", 0)
        h = matrix[cat].get("hard", 0)
        t = e + m + h
        flag = ""
        for diff, count in [("easy", e), ("medium", m), ("hard", h)]:
            if count < min_per_cell:
                flag = " ← GAP"
                gaps.append((cat, diff, count))
        print(f"{cat:<20} {e:>6} {m:>8} {h:>6} {t:>7}{flag}")

    totals = {"easy": 0, "medium": 0, "hard": 0}
    for cat, diffs in matrix.items():
        for d, c in diffs.items():
            totals[d] = totals.get(d, 0) + c
    grand = sum(totals.values())
    print("-" * 50)
    print(f"{'TOTAL':<20} {totals['easy']:>6} {totals['medium']:>8} {totals['hard']:>6} {grand:>7}")

    # Image group coverage
    print(f"\n{'=' * 60}")
    print("Image Group Coverage")
    print(f"{'=' * 60}")
    group_dist = {}
    for p in all_prompts:
        si = p.get("source_image", "")
        if "/person/" in si:
            g = "person"
        elif "/object/" in si:
            g = "object"
        elif "/scene/" in si:
            g = "scene"
        elif "/style/" in si:
            g = "style"
        else:
            g = "unknown"
        group_dist[g] = group_dist.get(g, 0) + 1
    for g in sorted(group_dist):
        print(f"  {g}: {group_dist[g]}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total prompts: {total}")
    print(f"Clean: {stats['clean']}")
    print(f"With warnings: {stats['warnings']}")
    print(f"Total warnings: {len(all_warnings)}")
    print(f"Coverage gaps: {len(gaps)}")
    if gaps:
        print(f"  Gaps: {gaps}")

    return len(all_warnings) == 0 and len(gaps) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Auto-reclassify mismatches")
    args = parser.parse_args()
    ok = main(args)
    if not ok:
        print("\n⚠ Validation found issues.")
    else:
        print("\n✓ All checks passed!")
