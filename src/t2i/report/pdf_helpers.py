"""PDF building block utilities (styles, table formatting)."""

from __future__ import annotations

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Table, TableStyle


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6))
    styles.add(
        ParagraphStyle(
            name="Disclosure",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            leading=10,
        )
    )
    return styles


def _df_to_table(df: pd.DataFrame, max_cols: int | None = None) -> Table:
    if max_cols:
        df = df.iloc[:, :max_cols]
    data = [list(df.columns)] + df.astype(object).values.tolist()
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def _leaderboard_display_df(lb: pd.DataFrame) -> pd.DataFrame:
    """Six-column leaderboard for the PDF: quality (covered) + full set (quality x reliability).

    Columns:
      model | AM (covered) | GM (covered) | AM (full) | GM (full) | coverage

    The covered columns are the apples-to-apples quality number a reviewer
    should quote; the full columns bake in the reliability discount. When
    the benchmark's `overall_*_covered` columns are absent (legacy run),
    covered collapses to full so the table still reads sensibly.
    """
    if "overall_gm" not in lb.columns or "overall_am" not in lb.columns:
        df = lb[["model", "overall_score", "n_prompts"]].copy()
        df = df.rename(columns={"overall_score": "score"})
        return df

    df = lb.copy()
    have_covered = "overall_gm_covered" in df.columns and "overall_am_covered" in df.columns
    if have_covered:
        df["AM (covered)"] = df["overall_am_covered"].fillna(df["overall_am"])
        df["GM (covered)"] = df["overall_gm_covered"].fillna(df["overall_gm"])
    else:
        df["AM (covered)"] = df["overall_am"]
        df["GM (covered)"] = df["overall_gm"]
    df["AM (full)"] = df["overall_am"]
    df["GM (full)"] = df["overall_gm"]

    if "n_covered" in df.columns and "n_total" in df.columns:
        df["coverage"] = (
            df["n_covered"].astype(int).astype(str) + "/" + df["n_total"].astype(int).astype(str)
        )
    else:
        df["coverage"] = (
            df["n_prompts"].astype(int).astype(str) + "/" + df["n_prompts"].astype(int).astype(str)
        )

    df = df.sort_values("GM (covered)", ascending=False).reset_index(drop=True)
    out = df[["model", "AM (covered)", "GM (covered)", "AM (full)", "GM (full)", "coverage"]]
    for c in ["AM (covered)", "GM (covered)", "AM (full)", "GM (full)"]:
        out[c] = out[c].map(lambda v: f"{v:.3f}")
    return out
