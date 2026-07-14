"""CLI: run image generation across selected models for all prompts.

Usage:
    python -m scripts.run_generation --models full        # default: 9-model production run
    python -m scripts.run_generation --models sanity      # 3-model pipeline check
    python -m scripts.run_generation --models all         # includes midjourney_v8
    python -m scripts.run_generation --models flux2_max,xai_aurora --layer 2
    python -m scripts.run_generation --dry-run
"""
import argparse
import asyncio
import os
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from src.t2i.generators import get_generator, all_registered
from src.t2i.generators.base import log_generation
from src.t2i.prompt_loader import load_prompt_set
from src.core.utils import CostTracker, get_logger
from src.t2i import OUTPUTS_DIR, load_models_config, load_settings

log = get_logger("run_generation")


def _seed_path(out_dir: Path, prompt_id: str, seed: int) -> Path:
    """Match the filename scheme used inside BaseGenerator.generate."""
    fname = f"{prompt_id}.png" if seed == 0 else f"{prompt_id}__s{seed}.png"
    return out_dir / fname


async def run_model(model_id: str, cfg: dict, prompts: list[dict],
                     cost: CostTracker, concurrency: int, seeds: int):
    out_dir = OUTPUTS_DIR / "generations" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    async with get_generator(model_id, cfg, cost, concurrency=concurrency) as gen:
        # For each (prompt, seed) pair, skip if the seed-specific PNG exists.
        # Seed 0 keeps the legacy filename so previous runs resume cleanly.
        tasks = []
        for p in prompts:
            for seed in range(seeds):
                if _seed_path(out_dir, p["prompt_id"], seed).exists():
                    continue
                tasks.append(gen.generate(
                    p["prompt_id"], p["prompt_text"], out_dir, seed=seed))
        if not tasks:
            log.info("[%s] all prompts x seeds already generated; skipping",
                     model_id)
            return
        results = await tqdm_asyncio.gather(*tasks, desc=f"gen:{model_id}")
        for r in results:
            log_generation(r)


async def main_async(args):
    prompts = load_prompt_set()
    if args.layer in (1, 2):
        prompts = [p for p in prompts if p["layer"] == args.layer]
    log.info("Loaded %d prompts (layer filter: %s)", len(prompts), args.layer)

    cfg_all = load_models_config()
    model_cfgs = cfg_all["models"]
    concurrency_map = cfg_all.get("concurrency", {})
    profiles = cfg_all.get("profiles", {})
    settings = load_settings()
    cap = float(os.getenv("MAX_TOTAL_COST_USD", settings["cost"]["hard_cap_usd"]))
    cost = CostTracker(hard_cap_usd=cap,
                        alert_at_fraction=settings["cost"]["alert_at_fraction"])

    # Seeds per prompt: CLI flag overrides settings.yaml. Seed 0 is the
    # legacy single-shot filename; seeds >=1 get the __s{N}.png suffix.
    default_seeds = int(settings.get("generation", {}).get("seeds_per_prompt", 1))
    seeds_n = max(1, int(args.seeds or default_seeds))

    if args.models in profiles:
        selected = list(profiles[args.models])
        log.info("Running profile '%s': %s", args.models, selected)
    else:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
        log.info("Running explicit models: %s", selected)

    available = set(all_registered())
    for m in selected:
        if m not in available:
            log.error("Model '%s' has no registered generator. Skipping.", m)
    selected = [m for m in selected if m in available]

    if args.dry_run:
        est = sum(model_cfgs[m]["cost_per_image"] * len(prompts) * seeds_n
                   for m in selected)
        total_imgs = len(prompts) * len(selected) * seeds_n
        print(f"Would generate {len(prompts)} prompts x {len(selected)} models "
              f"x {seeds_n} seeds = {total_imgs} images")
        print(f"Estimated cost: ${est:.2f} (cap ${cap})")
        return

    for model_id in selected:
        cfg = model_cfgs[model_id]
        conc = concurrency_map.get(cfg["provider"], 4)
        log.info("=== Running %s (provider=%s, concurrency=%d, seeds=%d) ===",
                 model_id, cfg["provider"], conc, seeds_n)
        await run_model(model_id, cfg, prompts, cost, conc, seeds_n)

    print("\nCost summary:")
    import json
    print(json.dumps(cost.summary(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="full",
                    help="Profile name ('sanity', 'full', 'all') or comma-separated model IDs")
    ap.add_argument("--layer", type=int, default=0, help="1, 2, or 0 (both)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Seeds per prompt (default: settings.yaml generation.seeds_per_prompt)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
