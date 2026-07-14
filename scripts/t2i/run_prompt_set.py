"""CLI: build the unified prompt set (Layer 1 + Layer 2).

Usage:
    python -m scripts.run_prompt_set                  # full build w/ Claude decomposition
    python -m scripts.run_prompt_set --skip-decomp    # skip decomposition (fast scaffold)
"""

import argparse

from src.t2i.prompt_loader import build_prompt_set, save_prompt_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-decomp",
        action="store_true",
        help="Skip Claude-based atomic decomposition (placeholder only)",
    )
    args = ap.parse_args()
    prompts = build_prompt_set(skip_decomposition=args.skip_decomp)
    path = save_prompt_set(prompts)
    print(f"Wrote {len(prompts)} prompts to {path}")


if __name__ == "__main__":
    main()
