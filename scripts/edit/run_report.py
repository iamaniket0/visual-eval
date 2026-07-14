"""CLI: generate the benchmark report from aggregated scores.

Usage:
    python -m scripts.run_report
    python -m scripts.run_report --format csv
    python -m scripts.run_report --format json

Reads from outputs/scores/ and prints a summary report. For the full PDF
report pipeline, see the T2I eval's report.py — this MVP emits a text/JSON
summary with the key tables.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.core.utils import get_logger
from src.edit import OUTPUTS_DIR

log = get_logger("run_report")

SCORES_DIR = OUTPUTS_DIR / "scores"


def load_if_exists(name: str) -> pd.DataFrame | None:
    path = SCORES_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["text", "csv", "json"], default="text")
    args = ap.parse_args()

    lb = load_if_exists("leaderboard.csv")
    psc = load_if_exists("per_subcategory.csv")
    pdim = load_if_exists("per_dimension.csv")
    lc = load_if_exists("layer_comparison.csv")
    fa = load_if_exists("failure_analysis.csv")
    fr = load_if_exists("filter_rates.csv")

    summary_path = SCORES_DIR / "summary_stats.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    if args.format == "json":
        report = {
            "summary": summary,
            "leaderboard": lb.to_dict(orient="records") if lb is not None else [],
            "per_subcategory": psc.to_dict(orient="records") if psc is not None else [],
            "per_dimension": pdim.to_dict(orient="records") if pdim is not None else [],
            "layer_comparison": lc.to_dict(orient="records") if lc is not None else [],
        }
        out_path = OUTPUTS_DIR / "reports" / "report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {out_path}")
        return

    print("=" * 70)
    print("IMAGE EDITING FAITHFULNESS BENCHMARK — REPORT")
    print("=" * 70)

    if summary:
        print(f"\nModels evaluated: {summary.get('n_models', '?')}")
        print(f"Prompts judged:   {summary.get('n_prompts_judged', '?')}")
        print(f"Overall AM:       {summary.get('mean_score_am_overall', '?')}")
        print(f"Overall GM:       {summary.get('mean_score_gm_overall', '?')}")

    if lb is not None:
        print("\n--- LEADERBOARD (by GM, covered prompts) ---")
        cols = [
            "model",
            "overall_gm_covered",
            "overall_am_covered",
            "n_covered",
            "n_total",
            "coverage_rate",
        ]
        avail = [c for c in cols if c in lb.columns]
        print(lb[avail].to_string(index=False))

    if pdim is not None and not pdim.empty:
        print("\n--- PER-DIMENSION SCORES ---")
        dim_cols = ["model"] + [c for c in pdim.columns if c.endswith("_gm") or c.endswith("_am")]
        avail = [c for c in dim_cols if c in pdim.columns]
        print(pdim[avail].to_string(index=False))

    if psc is not None and not psc.empty:
        print("\n--- PER-SUBCATEGORY (GM) ---")
        gm_cols = ["model"] + [c for c in psc.columns if c.endswith("__gm") or c == "overall_gm"]
        avail = [c for c in gm_cols if c in psc.columns]
        print(psc[avail].to_string(index=False))

    if lc is not None and not lc.empty:
        print("\n--- LAYER COMPARISON ---")
        print(lc.to_string(index=False))

    if fr is not None and not fr.empty:
        print("\n--- COVERAGE / FILTER RATES ---")
        print(fr.to_string(index=False))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
