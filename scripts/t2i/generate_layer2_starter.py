"""Generate a harder Layer 2 proprietary prompt set.

60 prompts = 20 per sub-category, with atomic binary decompositions pre-written.
For Complex Compositions, difficulty gradient matches the build doc:
  6 easy (3 constraints), 8 medium (4-5), 6 hard (6-8, pushed to 7-8).

Difficulty-bumping rules applied per internal guidance:

NUMERACY
  - baseline of 4+ objects (not 3 like T2I-CompBench++)
  - >= 10 prompts combine multiple counts (e.g. "5 red apples AND 3 green pears")
  - >= 6 prompts use counts >= 7 (20% more than the prior starter's 5)
  - >= 5 of the big-count (>=7) prompts bind a specific attribute to the count
  - >= 3 prompts use ordinal counting ("the 4th bottle from the left is blue")
  - every numeracy question explicitly says "exactly N"

SPATIAL_3D
  - >= 8 prompts use occlusion (object partially hidden behind another)
  - >= 5 prompts describe three depth planes (foreground / middle / background)
  - >= 3 prompts use viewer-relative perspective ("from the camera's viewpoint")
  - trivial "on top of" / "next to" relations deliberately de-emphasized in
    favor of behind / in front of / hidden by / partially occluded

COMPLEX_COMPOSITIONS
  - easy: each prompt must have >= 1 spatial OR numeracy constraint (not pure
    attribute binding)
  - medium: each prompt must have >= 1 numeracy AND >= 1 spatial constraint
  - hard: each prompt must have >= 2 numeracy, >= 1 spatial, >= 1 specific
    color attribute, and >= 1 material/texture attribute. 7-8 constraints
    is the norm here.

Question type enum is extended with "material" for questions about
texture/substance (e.g. "Is the surface wooden?"). Existing downstream code
(aggregator, judge, report) treats the type field as an opaque string so this
is a forward-compatible addition.

A runtime assertion enforces a minimum of 3 atomic questions per prompt
(guards against the L2_SP3_007-class regression).

Output: prompts/layer2_proprietary.json
"""

import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading — prompt data lives in layer2_data/*.json
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "layer2_data"


def _load_tuples(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    with open(path) as f:
        data = json.load(f)
    return [(d["text"], [tuple(q) for q in d["questions"]]) for d in data]


def _load_complex(path: Path) -> dict[str, list[tuple[str, list[tuple[str, str]]]]]:
    with open(path) as f:
        data = json.load(f)
    return {
        k: [(d["text"], [tuple(q) for q in d["questions"]]) for d in v] for k, v in data.items()
    }


NUMERACY = _load_tuples(_DATA_DIR / "numeracy.json")
SPATIAL_3D = _load_tuples(_DATA_DIR / "spatial.json")

_complex = _load_complex(_DATA_DIR / "complex.json")
COMPLEX_EASY = _complex["easy"]
COMPLEX_MEDIUM = _complex["medium"]
COMPLEX_HARD = _complex["hard"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TYPES = {"presence", "attribute", "numeracy", "spatial", "material"}


def _build(prompt_id, sub_cat, difficulty, text, questions):
    return {
        "prompt_id": prompt_id,
        "layer": 2,
        "sub_category": sub_cat,
        "difficulty": difficulty,
        "prompt_text": text,
        "atomic_questions": [
            {"q_id": f"q{i + 1}", "question": q[0], "type": q[1]} for i, q in enumerate(questions)
        ],
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "gold",
    "golden",
    "silver",
    "copper",
    "bronze",
}


def _count_type(questions, t):
    return sum(1 for q in questions if q["type"] == t)


def _numeracy_counts(prompt):
    """Return the list of integers extracted from 'exactly N' in numeracy qs."""
    nums = []
    for q in prompt["atomic_questions"]:
        if q["type"] != "numeracy":
            continue
        m = re.search(r"\bexactly\s+(\d+)", q["question"], re.IGNORECASE)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _is_ordinal(prompt):
    txt = prompt["prompt_text"].lower()
    ordinal_words = (
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    )
    if any(f"{w} " in txt for w in ordinal_words) and "from the" in txt:
        return True
    return bool(re.search(r"\b\d+(?:st|nd|rd|th)\b", txt))


def _has_color_attribute(prompt):
    """True if any atomic question names a specific color. We scan all types
    (not just 'attribute') because a numeracy question like
    'Are there exactly 3 red geraniums?' also binds a color constraint."""
    for q in prompt["atomic_questions"]:
        toks = re.findall(r"[a-z]+", q["question"].lower())
        if any(tok in _COLOR_WORDS for tok in toks):
            return True
    return False


def _validate_question_shape(prompts):
    """Enforce the minimum question count + valid type enum per prompt."""
    bad = []
    for p in prompts:
        n = len(p["atomic_questions"])
        if n < 3:
            bad.append(f"{p['prompt_id']} has only {n} questions (min 3)")
        for q in p["atomic_questions"]:
            if q["type"] not in VALID_TYPES:
                bad.append(
                    f"{p['prompt_id']} has invalid question type "
                    f"{q['type']!r} (valid: {sorted(VALID_TYPES)})"
                )
    if bad:
        raise AssertionError("Prompt validation failed:\n  " + "\n  ".join(bad))


def _validate_cohort_targets(prompts):
    """Enforce the per-sub-category difficulty targets documented above."""
    failures = []

    numeracy = [p for p in prompts if p["sub_category"] == "numeracy"]
    [p for p in prompts if p["sub_category"] == "spatial_3d"]
    cmp_easy = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "easy"
    ]
    cmp_medium = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "medium"
    ]
    cmp_hard = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "hard"
    ]

    # --- Numeracy targets ---
    # (a) every numeracy prompt's max exactly-N is >= 4
    for p in numeracy:
        nums = _numeracy_counts(p)
        if not nums:
            failures.append(f"{p['prompt_id']}: no 'exactly N' question")
        elif max(nums) < 4:
            failures.append(f"{p['prompt_id']}: max count {max(nums)} < 4 minimum")

    # (b) >= 10 compound prompts (2+ numeracy questions)
    compound = [p for p in numeracy if _count_type(p["atomic_questions"], "numeracy") >= 2]
    if len(compound) < 10:
        failures.append(f"numeracy: only {len(compound)} compound prompts, need >= 10")

    # (c) >= 6 big-count prompts (max count >= 7)
    big = [p for p in numeracy if _numeracy_counts(p) and max(_numeracy_counts(p)) >= 7]
    if len(big) < 6:
        failures.append(f"numeracy: only {len(big)} prompts with count >= 7, need >= 6")

    # (d) >= 5 big-count prompts that also bind a color/material attribute to the count
    big_with_attr = [
        p
        for p in big
        if _has_color_attribute(p) or _count_type(p["atomic_questions"], "material") >= 1
    ]
    if len(big_with_attr) < 5:
        failures.append(
            f"numeracy: only {len(big_with_attr)} big-count prompts with attribute binding, need >= 5"
        )

    # (e) >= 3 ordinal prompts
    ordinal = [p for p in numeracy if _is_ordinal(p)]
    if len(ordinal) < 3:
        failures.append(f"numeracy: only {len(ordinal)} ordinal prompts, need >= 3")

    # --- Complex easy: each must have >= 1 spatial or numeracy ---
    for p in cmp_easy:
        if (
            _count_type(p["atomic_questions"], "spatial") == 0
            and _count_type(p["atomic_questions"], "numeracy") == 0
        ):
            failures.append(f"{p['prompt_id']}: easy prompt has no spatial/numeracy constraint")

    # --- Complex medium: each must have >= 1 numeracy AND >= 1 spatial ---
    for p in cmp_medium:
        nq = _count_type(p["atomic_questions"], "numeracy")
        sq = _count_type(p["atomic_questions"], "spatial")
        if nq < 1 or sq < 1:
            failures.append(
                f"{p['prompt_id']}: medium prompt needs >=1 numeracy AND >=1 spatial "
                f"(got num={nq}, spat={sq})"
            )

    # --- Complex hard: >= 2 numeracy, >= 1 spatial, >= 1 color attr, >= 1 material ---
    for p in cmp_hard:
        nq = _count_type(p["atomic_questions"], "numeracy")
        sq = _count_type(p["atomic_questions"], "spatial")
        mq = _count_type(p["atomic_questions"], "material")
        cq = 1 if _has_color_attribute(p) else 0
        if nq < 2 or sq < 1 or mq < 1 or cq < 1:
            failures.append(
                f"{p['prompt_id']}: hard prompt needs >=2 num, >=1 spat, >=1 color-attr, >=1 material "
                f"(got num={nq}, spat={sq}, mat={mq}, color={cq})"
            )

    if failures:
        raise AssertionError("Cohort target validation failed:\n  " + "\n  ".join(failures))


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------


def generate_all():
    out = []
    for i, (text, qs) in enumerate(NUMERACY, start=1):
        out.append(_build(f"L2_NUM_{i:03d}", "numeracy", "medium", text, qs))
    for i, (text, qs) in enumerate(SPATIAL_3D, start=1):
        out.append(_build(f"L2_SP3_{i:03d}", "spatial_3d", "medium", text, qs))
    i = 1
    for text, qs in COMPLEX_EASY:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "easy", text, qs))
        i += 1
    for text, qs in COMPLEX_MEDIUM:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "medium", text, qs))
        i += 1
    for text, qs in COMPLEX_HARD:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "hard", text, qs))
        i += 1

    _validate_question_shape(out)
    _validate_cohort_targets(out)
    return out


def _print_numeracy_distribution(prompts):
    numeracy = [p for p in prompts if p["sub_category"] == "numeracy"]
    buckets = Counter()
    compound_count = 0
    big_with_attr_count = 0
    ordinal_count = 0
    for p in numeracy:
        nums = _numeracy_counts(p)
        mx = max(nums) if nums else 0
        if mx >= 7:
            buckets["7+"] += 1
        elif mx in (4, 5, 6):
            buckets[str(mx)] += 1
        else:
            buckets[f"<{4}"] += 1
        if _count_type(p["atomic_questions"], "numeracy") >= 2:
            compound_count += 1
        if mx >= 7 and (
            _has_color_attribute(p) or _count_type(p["atomic_questions"], "material") >= 1
        ):
            big_with_attr_count += 1
        if _is_ordinal(p):
            ordinal_count += 1
    print("Numeracy count distribution (max 'exactly N' per prompt):")
    for k in ("4", "5", "6", "7+"):
        print(f"  max = {k:<3}: {buckets.get(k, 0):2d} prompts")
    print(f"  compound (>= 2 numeracy questions):           {compound_count:2d} prompts")
    print(f"  big-count (>= 7) WITH attribute binding:      {big_with_attr_count:2d} prompts")
    print(f"  ordinal counting prompts:                     {ordinal_count:2d} prompts")


def main():
    prompts = generate_all()
    path = Path(__file__).resolve().parent.parent / "prompts" / "layer2_proprietary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {len(prompts)} prompts to {path}")
    c = Counter((p["sub_category"], p["difficulty"]) for p in prompts)
    for (sub, diff), n in sorted(c.items()):
        print(f"  {sub:25s} {diff:8s}: {n}")
    print()
    _print_numeracy_distribution(prompts)


if __name__ == "__main__":
    main()
