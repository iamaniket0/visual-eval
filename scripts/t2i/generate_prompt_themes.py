"""Extract a multi-label theme taxonomy from the prompt set and tag each prompt.

Two Claude calls via OpenRouter:

  1) One taxonomy-extraction call: all 210 prompt_text strings are sent in a
     single message; Claude returns 15-25 recurring, non-mutually-exclusive
     themes covering domain / setting / attribute / activity / composition
     axes.
  2) 210 per-prompt tagging calls, dispatched with asyncio and a concurrency
     cap of 8 to stay well inside OpenRouter rate limits. Each call returns
     2-5 theme IDs drawn from the taxonomy for that one prompt.

Outputs:
  prompts/theme_taxonomy.json   - [{id, description}, ...]
  prompts/prompt_themes.json    - {prompt_id: [theme_id, ...], ...}

Usage:
  python -m scripts.generate_prompt_themes

Completes in < 3 minutes on a 210-prompt set with the default concurrency.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from src.core.utils import get_api_key, get_logger
from src.t2i import PROMPTS_DIR, load_settings

log = get_logger("generate_prompt_themes")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://t2i-benchmark",
    "X-Title": "T2I Benchmark",
}

TAXONOMY_PATH = PROMPTS_DIR / "theme_taxonomy.json"
THEMES_PATH = PROMPTS_DIR / "prompt_themes.json"

CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TAXONOMY_SYSTEM = """You extract a compact, reusable theme taxonomy from a set \
of text-to-image prompts. The taxonomy will be used to slice evaluation scores \
(for example: "model X is worse at outdoor numeracy than indoor numeracy").

Hard requirements:
- Return BETWEEN 15 AND 25 themes. Not fewer, not more.
- Themes are multi-label / non-mutually-exclusive: a single prompt may be \
tagged with 2 to 5 themes.
- Each theme ID is short (1-3 words), lowercase, hyphenated if multi-word, \
and a STABLE SLUG (no spaces, no capitals, no punctuation beyond hyphens).
- Cover at least these axes (you may add more, but must cover these):
  * Domain:       animals, objects, people, food, vehicles, buildings, nature
  * Setting:      indoor, outdoor, urban, rural
  * Attribute:    color-heavy, material-heavy, size-contrast, age-related
  * Activity:     static, action, social-interaction
  * Composition: few-objects (<=3 items), medium-objects (4-6), dense (7+)

Return ONLY JSON in this exact shape (no prose, no fences):
{
  "themes": [
    {"id": "animals", "description": "one-line plain-English description"},
    {"id": "indoor", "description": "..."}
  ]
}"""

TAXONOMY_USER_TEMPLATE = """The prompt set (prompt_id: prompt_text, one per line):

{prompts_block}

Extract the theme taxonomy now."""

TAGGING_SYSTEM = """You tag a single text-to-image prompt with 2-5 applicable \
themes drawn from a fixed taxonomy.

Rules:
- Apply between 2 and 5 themes per prompt.
- Use ONLY theme IDs present in the provided taxonomy. Do not invent new IDs.
- Be literal. Infer themes from the explicit prompt content, not from \
speculation. If the prompt says "indoor", tag "indoor". Do not add themes for \
things the prompt does not mention.
- Return ONLY JSON (no prose, no fences):

{"themes": ["theme_id_1", "theme_id_2", "theme_id_3"]}"""

TAGGING_USER_TEMPLATE = """Taxonomy (JSON):
{taxonomy}

Prompt to tag:
"{prompt_text}"

Return the JSON tag list now."""


# ---------------------------------------------------------------------------
# JSON extraction (tolerant of fences / preamble)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# Taxonomy extraction (single call, all prompts)
# ---------------------------------------------------------------------------


async def extract_taxonomy(client, model: str, prompts: list[dict]) -> list[dict]:
    joined = "\n".join(f"- {p['prompt_id']}: {p['prompt_text']}" for p in prompts)
    resp = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=4000,
        extra_headers=OPENROUTER_HEADERS,
        messages=[
            {"role": "system", "content": TAXONOMY_SYSTEM},
            {"role": "user", "content": TAXONOMY_USER_TEMPLATE.format(prompts_block=joined)},
        ],
    )
    raw = resp.choices[0].message.content or ""
    parsed = _extract_json(raw)
    themes = parsed.get("themes", [])
    if not isinstance(themes, list):
        raise ValueError(f"Taxonomy payload missing 'themes' list: {parsed!r}")
    if not (15 <= len(themes) <= 25):
        log.warning("Taxonomy returned %d themes (spec: 15-25). Continuing.", len(themes))
    # Enforce minimal shape
    out = []
    for t in themes:
        tid = str(t.get("id", "")).strip().lower()
        if not tid:
            continue
        out.append({"id": tid, "description": str(t.get("description", "")).strip()})
    return out


# ---------------------------------------------------------------------------
# Per-prompt tagging (async, bounded concurrency)
# ---------------------------------------------------------------------------


async def tag_prompt(
    client,
    sema: asyncio.Semaphore,
    model: str,
    taxonomy_blob: str,
    prompt: dict,
    valid_ids: set[str],
) -> tuple[str, list[str]]:
    async with sema:
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=200,
                extra_headers=OPENROUTER_HEADERS,
                messages=[
                    {"role": "system", "content": TAGGING_SYSTEM},
                    {
                        "role": "user",
                        "content": TAGGING_USER_TEMPLATE.format(
                            taxonomy=taxonomy_blob, prompt_text=prompt["prompt_text"]
                        ),
                    },
                ],
            )
            raw = resp.choices[0].message.content or ""
            parsed = _extract_json(raw)
            tags = parsed.get("themes", [])
        except Exception as e:
            log.warning("Tagging failed for %s: %s", prompt["prompt_id"], e)
            return prompt["prompt_id"], []
    # Defensive: drop any tag not in taxonomy, keep order / dedup
    clean: list[str] = []
    seen: set[str] = set()
    for t in tags:
        tid = str(t).strip().lower()
        if tid in valid_ids and tid not in seen:
            clean.append(tid)
            seen.add(tid)
    return prompt["prompt_id"], clean


async def tag_all(
    client, model: str, taxonomy: list[dict], prompts: list[dict]
) -> dict[str, list[str]]:
    taxonomy_blob = json.dumps(
        [{"id": t["id"], "description": t["description"]} for t in taxonomy],
        indent=0,
    )
    valid_ids = {t["id"] for t in taxonomy}
    sema = asyncio.Semaphore(CONCURRENCY)
    tasks = [tag_prompt(client, sema, model, taxonomy_blob, p, valid_ids) for p in prompts]
    results = await asyncio.gather(*tasks)
    return dict(results)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_async():
    settings = load_settings()
    model_cfg = settings["atomic_decomposition"]["model"]
    model = model_cfg if "/" in model_cfg else f"anthropic/{model_cfg}"
    log.info("Using OpenRouter model: %s", model)

    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Themes require OpenRouter access.")

    prompts_path = PROMPTS_DIR / "prompt_set.json"
    if not prompts_path.exists():
        raise RuntimeError(f"{prompts_path} not found. Run scripts.run_prompt_set first.")
    with open(prompts_path) as f:
        prompts = json.load(f)
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    try:
        t0 = time.time()
        log.info("Extracting theme taxonomy...")
        taxonomy = await extract_taxonomy(client, model, prompts)
        log.info("Taxonomy: %d themes extracted in %.1fs", len(taxonomy), time.time() - t0)

        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TAXONOMY_PATH, "w") as f:
            json.dump(taxonomy, f, indent=2)
        log.info("Wrote %s", TAXONOMY_PATH)

        t1 = time.time()
        log.info("Tagging %d prompts (concurrency=%d)...", len(prompts), CONCURRENCY)
        tagged = await tag_all(client, model, taxonomy, prompts)
        log.info("Tagged %d prompts in %.1fs", len(tagged), time.time() - t1)
    finally:
        await client.close()

    # Diagnose + write results
    n_untagged = sum(1 for v in tagged.values() if not v)
    if n_untagged:
        log.warning("%d prompts received 0 themes (API/JSON failure).", n_untagged)
    tag_counts = [len(v) for v in tagged.values()]
    if tag_counts:
        log.info(
            "Tags per prompt: min=%d, max=%d, mean=%.1f",
            min(tag_counts),
            max(tag_counts),
            sum(tag_counts) / len(tag_counts),
        )

    with open(THEMES_PATH, "w") as f:
        json.dump(tagged, f, indent=2)
    log.info("Wrote %s", THEMES_PATH)

    # Compact summary line (handy when piped through tail)
    from collections import Counter

    theme_use = Counter(t for tags in tagged.values() for t in tags)
    print(f"Taxonomy: {len(taxonomy)} themes  |  Tagged: {len(tagged)} prompts")
    print("Top 10 themes by usage:")
    for tid, n in theme_use.most_common(10):
        print(f"  {tid:<22s}  {n:>4d} prompts")


def main():
    asyncio.run(run_async())


if __name__ == "__main__":
    main()
