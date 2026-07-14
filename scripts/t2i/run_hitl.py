"""CLI: HITL sampling, export, import, and agreement analysis.

Usage:
    python -m scripts.run_hitl sample            # build + save sample, export CSV
    python -m scripts.run_hitl import            # re-import the completed CSV
    python -m scripts.run_hitl score             # compute Cohen's kappa
    python -m scripts.run_hitl web               # launch Flask UI

Internal guidance specifies 10% stratified sample, Cohen's kappa > 0.6 target.
"""
import argparse

from src.t2i.hitl import build_sample, compute_agreement, export_csv, import_csv, save_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sample", "import", "score", "web"])
    ap.add_argument("--csv", help="CSV path for import (optional)")
    args = ap.parse_args()

    if args.action == "sample":
        sample = build_sample()
        save_sample(sample)
        path = export_csv(sample)
        print(f"Sampled {len(sample)} images. CSV: {path}")

    elif args.action == "import":
        path = import_csv(args.csv)
        print(f"Imported to: {path}")

    elif args.action == "score":
        result = compute_agreement()
        if result:
            print(f"Cohen's kappa: {result['cohen_kappa']}")
            print(f"Raw agreement: {result['raw_agreement']}")
            print(f"Target: > {result['target_kappa']}")
            print(f"n pairs: {result['n_pairs']}")

    elif args.action == "web":
        from src.t2i.hitl_webui import main as web_main
        web_main()


if __name__ == "__main__":
    main()
