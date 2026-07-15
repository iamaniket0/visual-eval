"""Stage 5: Report Generation (package)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.utils import get_logger
from src.t2i.report.aggregate import build_aggregate_report
from src.t2i.report.constants import SCORES_DIR
from src.t2i.report.model_card import build_model_card

log = get_logger("report")


def build_all_reports() -> list[Path]:
    out = [build_aggregate_report()]
    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")
    for model in lb["model"]:
        card = build_model_card(model)
        if card:
            out.append(card)
    return out


__all__ = ["build_all_reports", "build_aggregate_report", "build_model_card"]
