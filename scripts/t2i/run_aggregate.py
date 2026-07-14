"""CLI: aggregate scores into leaderboard + per-category CSVs + summary JSON."""
from src.t2i.aggregator import run_aggregation


def main():
    paths = run_aggregation()
    if not paths:
        print("No output produced - run the judge first.")
        return
    print("Wrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
