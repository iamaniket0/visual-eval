"""CLI: run image editing across selected models for all prompts.

Usage:
    python -m scripts.run_edit --models full
    python -m scripts.run_edit --models sanity
    python -m scripts.run_edit --models flux_kontext,bria_edit
    python -m scripts.run_edit --layer 2
    python -m scripts.run_edit --dry-run
"""

import argparse
import asyncio
import json
import os

from src.edit.editors import get_editor, all_registered
from src.edit.editors.base import log_edit
from src.edit.prompt_loader import load_all_prompts, resolve_source_image_path
from src.core.utils import CostTracker, get_logger
from src.edit import OUTPUTS_DIR, load_models_config, load_settings

log = get_logger("run_edit")


async def run_model(
    model_id: str, cfg: dict, prompts: list[dict], cost: CostTracker, concurrency: int
):
    out_dir = OUTPUTS_DIR / "edits" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    async with get_editor(model_id, cfg, cost, concurrency=concurrency) as editor:
        for p in prompts:
            prompt_id = p["prompt_id"]
            source_path = resolve_source_image_path(p)

            final_path = out_dir / f"{prompt_id}.png"
            if final_path.exists():
                continue

            turns = p.get("turns", 1)
            if isinstance(turns, list) and len(turns) > 1:
                if not editor.supports_multi_turn:
                    log.warning(
                        "[%s] skipping multi-turn prompt %s (not supported)", model_id, prompt_id
                    )
                    continue
                results = await editor.edit_multi_turn(
                    prompt_id=prompt_id,
                    source_image_path=source_path,
                    instructions=turns,
                    output_dir=out_dir,
                )
                for r in results:
                    log_edit(r)
            else:
                instruction = p.get("edit_instruction", "")
                result = await editor.edit(
                    prompt_id=prompt_id,
                    source_image_path=source_path,
                    edit_instruction=instruction,
                    output_dir=out_dir,
                )
                log_edit(result)

        n_done = len(list(out_dir.glob("*.png")))
        log.info("[%s] Coverage: %d/%d", model_id, n_done, len(prompts))


async def main_async(args):
    prompts = load_all_prompts()
    if args.layer in (1, 2):
        prompts = [p for p in prompts if p["layer"] == args.layer]

    if args.prompt_ids:
        ids = set(args.prompt_ids.split(","))
        prompts = [p for p in prompts if p["prompt_id"] in ids]
    elif args.num_prompts and args.num_prompts < len(prompts):
        cats = sorted(set(p["sub_category"] for p in prompts))
        picked: list[dict] = []
        per_cat = max(1, args.num_prompts // len(cats))
        for cat in cats:
            bucket = [p for p in prompts if p["sub_category"] == cat]
            picked.extend(bucket[:per_cat])
        remaining = args.num_prompts - len(picked)
        rest = [p for p in prompts if p not in picked]
        picked.extend(rest[: max(0, remaining)])
        prompts = picked[: args.num_prompts]

    log.info("Loaded %d prompts (layer filter: %s)", len(prompts), args.layer)

    cfg_all = load_models_config()
    model_cfgs = cfg_all["models"]
    concurrency_map = cfg_all.get("concurrency", {})
    profiles = cfg_all.get("profiles", {})
    settings = load_settings()
    cap = float(os.getenv("MAX_TOTAL_COST_USD", settings["cost"]["hard_cap_usd"]))
    cost = CostTracker(hard_cap_usd=cap, alert_at_fraction=settings["cost"]["alert_at_fraction"])

    if args.models in profiles:
        selected = list(profiles[args.models])
        log.info("Running profile '%s': %s", args.models, selected)
    else:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
        log.info("Running explicit models: %s", selected)

    available = set(all_registered())
    for m in selected:
        if m not in available:
            log.error("Model '%s' has no registered editor. Skipping.", m)
    selected = [m for m in selected if m in available]

    if args.dry_run:
        est = sum(model_cfgs[m]["cost_per_edit"] * len(prompts) for m in selected)
        total = len(prompts) * len(selected)
        print(f"Would edit {len(prompts)} prompts x {len(selected)} models = {total} edits")
        print(f"Estimated cost: ${est:.2f} (cap ${cap})")
        return

    for model_id in selected:
        cfg = model_cfgs[model_id]
        conc = concurrency_map.get(cfg["provider"], 4)
        log.info(
            "=== Running %s (provider=%s, concurrency=%d) ===", model_id, cfg["provider"], conc
        )
        await run_model(model_id, cfg, prompts, cost, conc)

    print("\nCost summary:")
    print(json.dumps(cost.summary(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="full",
        help="Profile name ('sanity', 'full') or comma-separated model IDs",
    )
    ap.add_argument("--layer", type=int, default=0, help="1, 2, or 0 (both)")
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit to N prompts (balanced across sub-categories)",
    )
    ap.add_argument(
        "--prompt-ids",
        type=str,
        default=None,
        help="Comma-separated prompt IDs to run (overrides --num-prompts)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
