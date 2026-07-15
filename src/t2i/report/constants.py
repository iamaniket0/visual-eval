"""Shared constants for the report package."""

from __future__ import annotations

from src.t2i import OUTPUTS_DIR

REPORTS_DIR = OUTPUTS_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
SCORES_DIR = OUTPUTS_DIR / "scores"

DISCLOSURE_LAYERS = (
    "Layer 1 prompts are drawn from T2I-CompBench++ (NeurIPS 2023, TPAMI 2025), "
    "a public benchmark. Frontier models may have been calibrated against these "
    "prompts during training. Layer 2 uses proprietary, unpublished prompts to "
    "control for this."
)

# Minimum prompt count for a theme to appear in PDF displays.
THEME_MIN_N = 15
THEME_FILTER_NOTE = (
    f"Themes with fewer than {THEME_MIN_N} prompts per model are excluded "
    "from the chart and top/bottom lists: below that threshold the binomial "
    "95% CI widens past ±0.13 and per-prompt noise dominates the theme mean. "
    "Low-n themes remain in the raw theme_breakdown.csv for debugging."
)

# Color palette - GM gets the saturated accent, AM the muted secondary.
GM_COLOR = "#4a90e2"  # primary blue
AM_COLOR = "#b0c9e4"  # desaturated blue
