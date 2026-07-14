"""Stage 1: Prompt Loading.

Loads Layer 1 (T2I-CompBench++) and Layer 2 (internal proprietary) prompts,
generates atomic binary decompositions for Layer 1 via Claude, and writes
a unified prompt set to prompts/prompt_set.json.

Prompt schema (stored as JSON):
    {
      "prompt_id": "L1_NUM_001",
      "layer": 1,
      "sub_category": "numeracy",
      "difficulty": "medium",
      "prompt_text": "...",
      "atomic_questions": [
        {"q_id": "q1", "question": "...", "type": "presence|attribute|numeracy|spatial"}
      ]
    }
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from src.core.utils import ROOT, get_logger, get_api_key
from src.t2i import PROMPTS_DIR, load_settings

log = get_logger("prompt_loader")

T2I_COMPBENCH_REPO = "https://github.com/Karine-Huang/T2I-CompBench.git"
T2I_COMPBENCH_LOCAL = ROOT / "T2I-CompBench"

SUB_CATEGORIES = ["numeracy", "complex_compositions", "spatial_3d"]


# ---------------------------------------------------------------------------
# Layer 1: T2I-CompBench++ extraction
# ---------------------------------------------------------------------------


def ensure_compbench_cloned() -> Path:
    """Clone T2I-CompBench if not already present locally."""
    if T2I_COMPBENCH_LOCAL.exists():
        log.info("T2I-CompBench already cloned at %s", T2I_COMPBENCH_LOCAL)
        return T2I_COMPBENCH_LOCAL
    log.info("Cloning T2I-CompBench...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", T2I_COMPBENCH_REPO, str(T2I_COMPBENCH_LOCAL)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log.error("Failed to clone T2I-CompBench: %s", e.stderr)
        raise
    return T2I_COMPBENCH_LOCAL


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def load_compbench_raw(sub_category: str) -> list[str]:
    """Load raw prompts for a sub-category from T2I-CompBench repo.

    Paths in the repo have shifted over time, so we probe a few candidates
    and parse whatever format we find (txt = one prompt per line).
    """
    repo = T2I_COMPBENCH_LOCAL
    if not repo.exists():
        log.warning("T2I-CompBench not cloned; returning empty list for %s", sub_category)
        return []

    candidates: list[Path] = []
    if sub_category == "numeracy":
        candidates = [
            repo / "examples/dataset/numeracy_val.txt",
            repo / "examples/dataset/numeracy.txt",
            repo / "examples/dataset/numeracy_train.txt",
        ]
    elif sub_category == "complex_compositions":
        candidates = [
            repo / "examples/dataset/complex_val.txt",
            repo / "examples/dataset/complex.txt",
        ]
    elif sub_category == "spatial_3d":
        candidates = [
            repo / "examples/dataset/spatial_3d_val.txt",
            repo / "examples/dataset/3d_spatial_val.txt",
            repo / "UniDet_eval/3d_spatial_val.txt",
            repo / "examples/dataset/spatial_val.txt",  # fallback to 2D if 3D missing
        ]

    path = _find_first_existing(candidates)
    if path is None:
        log.warning("No prompt file found for %s; tried: %s", sub_category, candidates)
        return []

    log.info("Loading %s from %s", sub_category, path.relative_to(repo))
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def stratified_sample(prompts: list[str], n: int, seed: int = 42) -> list[str]:
    """Stratified random sample by prompt length as a difficulty proxy.

    Splits into length terciles (short/medium/long) and samples proportionally.
    For MVP this is a reasonable difficulty spread without needing the full
    T2I-CompBench difficulty metadata, which isn't uniformly available across
    sub-categories.
    """
    if len(prompts) <= n:
        return prompts.copy()
    rng = random.Random(seed)
    by_len = sorted(prompts, key=len)
    tercile = len(by_len) // 3
    short = by_len[:tercile]
    med = by_len[tercile : 2 * tercile]
    long = by_len[2 * tercile :]
    per_bucket = n // 3
    remainder = n - per_bucket * 3
    samples = (
        rng.sample(short, min(per_bucket, len(short)))
        + rng.sample(med, min(per_bucket, len(med)))
        + rng.sample(long, min(per_bucket + remainder, len(long)))
    )
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Atomic decomposition via Claude
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM = """You decompose text-to-image prompts into atomic binary (yes/no) \
questions for evaluating image faithfulness.

Rules:
- Each question must be answerable with ONLY yes or no from looking at the image.
- Keep questions simple: presence, attribute, count, or relative position.
- NO layered rubrics, NO multi-step judgments, NO quality judgments.
- Number of questions: 3-7 depending on prompt complexity.
- Classify each as one of: presence, attribute, numeracy, spatial.

Return ONLY JSON in this exact shape:
{
  "questions": [
    {"question": "Are there apples in the image?", "type": "presence"},
    {"question": "Are the apples red?", "type": "attribute"},
    {"question": "Are there exactly 5 apples?", "type": "numeracy"}
  ]
}"""

DECOMPOSITION_USER_TEMPLATE = (
    'Prompt: "{prompt}"\nSub-category: {sub_category}\n\nDecompose into atomic binary questions.'
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from model response, handling code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://visual-eval-benchmark",
    "X-Title": "visual-eval-benchmark",
}


def _decompose_via_openrouter(
    model: str, prompt_text: str, sub_category: str
) -> list[dict[str, Any]]:
    """Call an Anthropic model through OpenRouter's OpenAI-compatible API."""
    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        log.warning(
            "OPENROUTER_API_KEY not set; using placeholder decomposition for: %s", prompt_text[:60]
        )
        return []
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    or_model = model if "/" in model else f"anthropic/{model}"
    resp = client.chat.completions.create(
        model=or_model,
        temperature=0.0,
        max_tokens=800,
        extra_headers=OPENROUTER_HEADERS,
        messages=[
            {"role": "system", "content": DECOMPOSITION_SYSTEM},
            {
                "role": "user",
                "content": DECOMPOSITION_USER_TEMPLATE.format(
                    prompt=prompt_text,
                    sub_category=sub_category,
                ),
            },
        ],
    )
    raw = resp.choices[0].message.content or ""
    return _extract_json(raw).get("questions", [])


def _decompose_via_anthropic(
    model: str, prompt_text: str, sub_category: str
) -> list[dict[str, Any]]:
    """Call Claude directly via the native Anthropic SDK."""
    api_key = get_api_key("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set; using placeholder decomposition for: %s", prompt_text[:60]
        )
        return []
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.0,
        system=DECOMPOSITION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": DECOMPOSITION_USER_TEMPLATE.format(
                    prompt=prompt_text,
                    sub_category=sub_category,
                ),
            }
        ],
    )
    raw = resp.content[0].text
    return _extract_json(raw).get("questions", [])


def generate_decomposition(prompt_text: str, sub_category: str) -> list[dict[str, str]]:
    """Generate atomic binary questions for a prompt.

    Backend chosen via settings.api_routing.decomposition:
      - "openrouter" -> Claude via OpenRouter's OpenAI-compatible API
      - "anthropic"  -> native Anthropic SDK
    Both paths feed the same _extract_json parser. On any failure or missing
    key, falls back to a deterministic placeholder so the pipeline stays
    runnable during scaffolding.
    """
    settings = load_settings()
    model = settings["atomic_decomposition"]["model"]
    routing = settings.get("api_routing", {}).get("decomposition", "anthropic")

    try:
        if routing == "openrouter":
            questions = _decompose_via_openrouter(model, prompt_text, sub_category)
        else:
            questions = _decompose_via_anthropic(model, prompt_text, sub_category)
    except Exception as e:
        log.warning(
            "Decomposition (%s) failed for '%s': %s. Using placeholder.",
            routing,
            prompt_text[:60],
            e,
        )
        return _placeholder_decomposition(prompt_text)

    if not questions:
        return _placeholder_decomposition(prompt_text)

    out = []
    for i, q in enumerate(questions, start=1):
        out.append(
            {
                "q_id": f"q{i}",
                "question": q["question"],
                "type": q.get("type", "presence"),
            }
        )
    return out


def _placeholder_decomposition(prompt_text: str) -> list[dict[str, str]]:
    """Lightweight fallback used only when Claude is unavailable."""
    return [
        {
            "q_id": "q1",
            "question": f'Does the image match the prompt: "{prompt_text}"?',
            "type": "presence",
        },
    ]


# ---------------------------------------------------------------------------
# Layer 2: proprietary prompts
# ---------------------------------------------------------------------------


def load_layer2_proprietary() -> list[dict[str, Any]]:
    path = PROMPTS_DIR / "layer2_proprietary.json"
    if not path.exists():
        log.warning("%s not found. Run scripts/generate_layer2_starter.py first.", path)
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _prompt_id(layer: int, sub_category: str, idx: int) -> str:
    code = {
        "numeracy": "NUM",
        "complex_compositions": "CMP",
        "spatial_3d": "SP3",
    }[sub_category]
    return f"L{layer}_{code}_{idx:03d}"


def build_prompt_set(skip_decomposition: bool = False) -> list[dict[str, Any]]:
    """Build the full prompt set across layer 1 and layer 2."""
    settings = load_settings()
    n_l1 = settings["prompt_sampling"]["layer1_per_subcategory"]

    ensure_compbench_cloned()
    all_prompts: list[dict[str, Any]] = []

    # Layer 1
    for sub in SUB_CATEGORIES:
        raw = load_compbench_raw(sub)
        if not raw:
            log.warning("Layer 1 sub-category '%s' has no prompts; skipping", sub)
            continue
        sampled = stratified_sample(raw, n_l1)
        log.info("Layer 1 %s: sampled %d / %d prompts", sub, len(sampled), len(raw))
        for i, text in enumerate(sampled, start=1):
            pid = _prompt_id(1, sub, i)
            questions = [] if skip_decomposition else generate_decomposition(text, sub)
            all_prompts.append(
                {
                    "prompt_id": pid,
                    "layer": 1,
                    "sub_category": sub,
                    "difficulty": "auto",
                    "prompt_text": text,
                    "atomic_questions": questions,
                }
            )

    # Layer 2
    l2 = load_layer2_proprietary()
    for item in l2:
        # Trust any decomposition that came with the Layer 2 file.
        all_prompts.append(item)

    log.info(
        "Built prompt set: %d total (%d Layer 1, %d Layer 2)",
        len(all_prompts),
        sum(1 for p in all_prompts if p["layer"] == 1),
        sum(1 for p in all_prompts if p["layer"] == 2),
    )
    return all_prompts


def save_prompt_set(prompts: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or (PROMPTS_DIR / "prompt_set.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
    log.info("Wrote %d prompts to %s", len(prompts), path)
    return path


def load_prompt_set(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (PROMPTS_DIR / "prompt_set.json")
    with open(path) as f:
        return json.load(f)
