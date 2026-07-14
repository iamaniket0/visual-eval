"""CLI: aggregate judgment scores into leaderboards and breakdowns.

Usage:
    python -m scripts.run_aggregate
"""

from src.edit.aggregator import run_aggregation
from src.core.utils import get_logger

log = get_logger("run_aggregate")


def main():
    paths = run_aggregation()
    if paths:
        print("\nAggregation outputs:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print("No aggregation output — run the judge first.")


if __name__ == "__main__":
    main()
