"""CLI: build all PDF reports (aggregate + per-model cards).

Usage:
    python -m scripts.run_report            # aggregate + all cards
    python -m scripts.run_report --model flux2_max   # single card
"""
import argparse

from src.t2i.report import build_aggregate_report, build_all_reports, build_model_card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Build a single model card only")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.model:
        path = build_model_card(args.model)
        if path:
            print(f"Wrote: {path}")
        return
    if args.aggregate_only:
        path = build_aggregate_report()
        print(f"Wrote: {path}")
        return
    for path in build_all_reports():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
