"""Replicate R2I-Bench's own evaluation system (R2I-Score) on our generated images.

Uses GPT-4o via OpenRouter as the judge (matching their methodology).
Produces per-category R2I-Scores directly comparable to their Table 3.

Usage:
    python -m scripts.run_r2i_eval
    python -m scripts.run_r2i_eval --model lucid_origin
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.core.utils import get_logger
from src.t2i import OUTPUTS_DIR

load_dotenv()
log = get_logger("r2i_eval")

R2I_DIR = Path("external/R2I-Bench")
PROMPTS_DIR = R2I_DIR / "data" / "prompts"
EVAL_DIR = R2I_DIR / "data" / "evaluation"

JUDGE_PROMPT = """
# Text-to-Image Quality Evaluation Protocol
## System Instruction
You are an AI quality auditor for text-to-image generation. Answer these questions with ABSOLUTE RUTHLESSNESS.
Only images meeting the HIGHEST standards should receive top scores.

## Task Overview
The image is generated from the prompt:
[PROMPT]

## Question List
[QUESTION_LIST]

## Output Format
Analyze the image, then output ONLY a JSON block with question IDs as keys and scores (0.0, 0.5, or 1.0) as values.

```json
{
    "id": score,
    ...
}
```
"""


def get_llm_response(prompt: str, image_path: str) -> str:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        temperature=0.1,
        model="openai/gpt-4o",
        extra_headers={"HTTP-Referer": "https://t2i-benchmark", "X-Title": "T2I Benchmark"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content


CATEGORY_SUBCATEGORY_LIST = [
    {"category": "causal", "subcategory": "cause_to_effect"},
    {"category": "causal", "subcategory": "effect_to_cause"},
    {"category": "numerical", "subcategory": "approximate_number_generation"},
    {"category": "numerical", "subcategory": "conceptual_quantitative"},
    {"category": "numerical", "subcategory": "exact_number_generation"},
    {"category": "compositional", "subcategory": "creative_compositional"},
    {"category": "compositional", "subcategory": "inferential_spatial"},
    {"category": "compositional", "subcategory": "prescriptive_spatial"},
    {"category": "logical", "subcategory": "abductive"},
    {"category": "logical", "subcategory": "categorical"},
    {"category": "logical", "subcategory": "conjunctive"},
    {"category": "logical", "subcategory": "deductive"},
    {"category": "logical", "subcategory": "disjunctive"},
    {"category": "logical", "subcategory": "hypothetical"},
    {"category": "logical", "subcategory": "sufficient_conditional"},
    {"category": "commonsense", "subcategory": "affordance"},
    {"category": "commonsense", "subcategory": "attribute"},
    {"category": "commonsense", "subcategory": "color"},
    {"category": "commonsense", "subcategory": "emotion_intention_commonsense"},
    {"category": "commonsense", "subcategory": "social_cultural_knowledge_object"},
    {"category": "commonsense", "subcategory": "social_cultural_knowledge_scene"},
    {"category": "commonsense", "subcategory": "temporal_understanding"},
    {"category": "concept_mixing", "subcategory": "functional_mixing"},
    {"category": "concept_mixing", "subcategory": "literal_mixing"},
]


def find_image(model: str, category: str, subcategory: str, source_id: int) -> str | None:
    """Find the generated image for a given R2I prompt.

    Our images are named R2I_{CAT}_{NNN}.png where CAT is the 3-letter
    category prefix and NNN is sequential. We need to map back from
    (category, subcategory, source_id) to our prompt_id.
    """
    gen_dir = OUTPUTS_DIR / "generations" / model
    prompt_csv = PROMPTS_DIR / category / f"{category}_{subcategory}.csv"

    if not prompt_csv.exists():
        return None

    prompts_map = json.load(open("prompts/prompt_set.json"))
    r2i_prompts = {
        p["prompt_text"]: p["prompt_id"] for p in prompts_map if p["prompt_id"].startswith("R2I_")
    }

    with open(prompt_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("id", 0)) == source_id:
                prompt_text = row.get("Prompt", "")
                pid = r2i_prompts.get(prompt_text)
                if pid:
                    img = gen_dir / f"{pid}.png"
                    if img.exists():
                        return str(img)
    return None


def run_r2i_eval(model: str) -> dict[str, list[float]]:
    """Run R2I-Bench evaluation on a single model. Returns {category: [scores]}."""
    category_scores = defaultdict(list)

    for entry in CATEGORY_SUBCATEGORY_LIST:
        category = entry["category"]
        subcategory = entry["subcategory"]

        eval_csv = EVAL_DIR / category / f"{subcategory}_eval.csv"
        prompt_csv = PROMPTS_DIR / category / f"{category}_{subcategory}.csv"

        if not eval_csv.exists() or not prompt_csv.exists():
            continue

        eval_df = list(csv.DictReader(open(eval_csv)))
        prompt_rows = {int(r["id"]): r for r in csv.DictReader(open(prompt_csv))}

        grouped = defaultdict(list)
        for row in eval_df:
            grouped[int(row["source_id"])].append(row)

        for source_id, questions in grouped.items():
            if source_id not in prompt_rows:
                continue

            item_prompt = prompt_rows[source_id]["Prompt"]
            image_path = find_image(model, category, subcategory, source_id)

            if not image_path:
                continue

            q_list_str = "\n".join(
                f"{q['id']}. {q['question']} \nCriteria: {q['evaluation_criteria']}"
                for q in questions
            )

            final_prompt = JUDGE_PROMPT.replace("[PROMPT]", item_prompt).replace(
                "[QUESTION_LIST]", q_list_str
            )

            try:
                evaluation = get_llm_response(final_prompt, image_path)

                json_match = re.search(r"```json\n(.*?)\n```", evaluation, flags=re.DOTALL)
                json_str = json_match.group(1) if json_match else evaluation
                fixed_str = re.sub(r"[\x00-\x1f\x7f]", "", json_str)
                evaluation_dict = json.loads(fixed_str)

                above = 0
                under = 0
                for q in questions:
                    q_id = str(q["id"])
                    weight = float(q["weight"])
                    if q_id in evaluation_dict:
                        score = float(evaluation_dict[q_id])
                        above += weight * score
                        under += weight

                final_score = round(above / under, 2) if under > 0 else 0.0
                category_scores[category].append(final_score)
                log.info("%s/%s id=%d score=%.2f", category, subcategory, source_id, final_score)

            except Exception as e:
                log.warning("Error on %s/%s id=%d: %s", category, subcategory, source_id, e)

    return dict(category_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    models = (
        [args.model]
        if args.model
        else [
            "lucid_origin",
            "xai_aurora",
            "gpt_image_2",
            "flux2_max",
            "bria_fibo",
            "gpt_image_15",
        ]
    )

    results = {}
    for model in models:
        log.info("=== Running R2I-Score eval on %s ===", model)
        scores = run_r2i_eval(model)
        results[model] = {}
        for cat, cat_scores in scores.items():
            avg = round(sum(cat_scores) / len(cat_scores), 2) if cat_scores else 0
            results[model][cat] = {"avg": avg, "n": len(cat_scores)}
            log.info("%s %s: avg=%.2f n=%d", model, cat, avg, len(cat_scores))

    out_path = OUTPUTS_DIR / "scores" / "r2i_score_results.json"
    json.dump(results, open(out_path, "w"), indent=2)
    log.info("Wrote R2I-Score results to %s", out_path)

    print("\n=== R2I-Score Results (GPT-4o judge, comparable to Table 3) ===\n")
    cats = ["causal", "numerical", "logical", "compositional", "commonsense", "concept_mixing"]
    header = f"{'model':24s}" + "".join(f" {c[:8]:>8s}" for c in cats)
    print(header)
    print("─" * 80)
    for model in models:
        line = f"{model:24s}"
        for cat in cats:
            r = results.get(model, {}).get(cat, {})
            if r:
                line += f" {r['avg']:8.2f}"
            else:
                line += f" {'---':>8s}"
        print(line)


if __name__ == "__main__":
    main()
