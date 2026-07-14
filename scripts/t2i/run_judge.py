"""CLI: run the MLLM judge on generated images.

Usage:
    python -m scripts.run_judge                            # settings.yaml judge.backend
    python -m scripts.run_judge --models sanity
    python -m scripts.run_judge --models all               # includes midjourney_v8
    python -m scripts.run_judge --models flux2_max
    python -m scripts.run_judge --backend gpt4o_soft       # override settings backend
    python -m scripts.run_judge --backend gpt4o_hard       # reproduce pre-migration runs
    python -m scripts.run_judge --backend qwen_soft        # blocked on current OpenRouter

See `src/judge.py` for the three judge backends and the Soft-TIFA rationale.
"""

import argparse
import asyncio
import json

from src.t2i.judge import judge_model_generations
from src.t2i.prompt_loader import load_prompt_set
from src.core.utils import CostTracker, get_logger
from src.t2i import load_models_config, load_settings

log = get_logger("run_judge")


async def main_async(args):
    prompts = {p["prompt_id"]: p for p in load_prompt_set()}
    settings = load_settings()
    cost = CostTracker(hard_cap_usd=settings["cost"]["hard_cap_usd"])

    cfg_all = load_models_config()
    profiles = cfg_all.get("profiles", {})
    if args.models in profiles:
        model_ids = list(profiles[args.models])
        log.info("Running profile '%s': %s", args.models, model_ids)
    else:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        log.info("Running explicit models: %s", model_ids)

    # Backend override comes from CLI or settings.yaml. Display name follows
    # the legacy --judge flag if provided (reportable in JudgeResult records).
    backend = args.backend or settings.get("judge", {}).get("backend", "gpt4o_hard")
    log.info("Judge backend: %s", backend)

    for mid in model_ids:
        log.info("=== Judging %s with backend=%s ===", mid, backend)
        await judge_model_generations(
            mid,
            prompts,
            cost,
            judge_model=args.judge,
            backend=backend,
        )

    print("\nCost summary:")
    print(json.dumps(cost.summary(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="full",
        help="Profile name ('sanity', 'full', 'all') or comma-separated model IDs",
    )
    ap.add_argument(
        "--judge",
        default=None,
        help="Explicit display-name override for the judge identity "
        "in JSONL records. Omit to use the backend's real identity "
        "(recommended - prevents provenance labels from drifting "
        "off the actual backend).",
    )
    ap.add_argument(
        "--backend",
        default=None,
        choices=[None, "gpt4o_hard", "gpt4o_soft", "qwen_soft", "qwen_together_soft"],
        help="Judge backend override (default: settings.yaml judge.backend)",
    )
    args = ap.parse_args()
    # Previously defaulted to settings.judge.primary ("gpt-4o"), which silently
    # relabeled every Qwen judgment as gpt-4o. Leave unset now so the backend
    # keeps its real identity (Qwen/Qwen3.5-397B-A17B for qwen_together_soft,
    # etc.). Users who want a display override can still pass --judge.
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
