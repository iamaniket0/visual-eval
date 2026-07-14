"""CLI: run the MLLM judge on edited images.

Usage:
    python -m scripts.run_judge
    python -m scripts.run_judge --models sanity
    python -m scripts.run_judge --models flux_kontext,bria_edit
    python -m scripts.run_judge --backend qwen_together_soft

The judge receives BOTH the source image AND the edited image to evaluate
instruction following, visual consistency, and detail preservation.
"""

import argparse
import asyncio
import json

from src.core.utils import CostTracker, get_logger
from src.edit import load_models_config, load_settings
from src.edit.judge import judge_model_edits
from src.edit.prompt_loader import prompts_by_id

log = get_logger("run_judge")


async def main_async(args):
    pmap = prompts_by_id()
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

    backend = args.backend or settings.get("judge", {}).get("backend", "qwen_together_soft")
    log.info("Judge backend: %s", backend)

    for mid in model_ids:
        log.info("=== Judging %s with backend=%s ===", mid, backend)
        await judge_model_edits(mid, pmap, cost, backend=backend)

    print("\nCost summary:")
    print(json.dumps(cost.summary(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="full", help="Profile name or comma-separated model IDs")
    ap.add_argument(
        "--backend",
        default=None,
        choices=[None, "qwen_together_soft"],
        help="Judge backend override",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
