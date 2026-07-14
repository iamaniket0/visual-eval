"""Stage 4: Score aggregation with per-dimension breakdown.

Reads outputs/judgments/{model}.jsonl and produces:
    outputs/scores/leaderboard.csv
    outputs/scores/per_subcategory.csv
    outputs/scores/per_dimension.csv          <- NEW: instruction_following / visual_consistency / detail_preservation
    outputs/scores/layer_comparison.csv
    outputs/scores/failure_analysis.csv
    outputs/scores/filter_rates.csv
    outputs/scores/summary_stats.json

Scoring: Soft-TIFA AM/GM (Kamath et al., GenEval 2, 2025).
Dimensions map to GEditBench v2's three-axis evaluation:
    - instruction_following: did the requested edit happen?
    - visual_consistency: are unedited regions unchanged?
    - detail_preservation: are fine details (text, textures, edges) intact?
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils import get_logger, read_jsonl
from src.core.scoring import (
    soft_tifa_am,
    soft_tifa_gm,
    probabilities_from_answers,
    DEFAULT_LOGPROB_FLOOR,
)
from src.edit import OUTPUTS_DIR, PROMPTS_DIR

log = get_logger("aggregator")

SCORES_DIR = OUTPUTS_DIR / "scores"

DIMENSIONS = ["instruction_following", "visual_consistency", "detail_preservation"]

# Alias for local usage — matches the original private name
_probabilities_from_answers = probabilities_from_answers


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_prompt_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Load prompts from both layer files."""
    prompts: list[dict[str, Any]] = []
    for fname in ["layer1_gold.json", "layer2_proprietary.json"]:
        p = (path or PROMPTS_DIR) / fname if path and path.is_dir() else PROMPTS_DIR / fname
        if p.exists():
            with open(p) as f:
                prompts.extend(json.load(f))
    return prompts


def _load_all_judgments() -> pd.DataFrame:
    rows = []
    for path in (OUTPUTS_DIR / "judgments").glob("*.jsonl"):
        model = path.stem
        for rec in read_jsonl(path):
            answers = rec.get("answers", []) or []
            score_am = rec.get("score_am")
            score_gm = rec.get("score_gm")
            if score_am is None or score_gm is None:
                probs = _probabilities_from_answers(answers)
                score_am = soft_tifa_am(probs) if score_am is None else float(score_am)
                score_gm = soft_tifa_gm(probs) if score_gm is None else float(score_gm)
            rows.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "model": model,
                    "score_am": float(score_am),
                    "score_gm": float(score_gm),
                    "judge_error": rec.get("error"),
                    "answers": answers,
                }
            )
    if not rows:
        log.warning("No judgments found")
        return pd.DataFrame(
            columns=["prompt_id", "model", "score_am", "score_gm", "judge_error", "answers"]
        )
    return pd.DataFrame(rows)


def _merge(judgments: pd.DataFrame, prompts: list[dict]) -> pd.DataFrame:
    prompt_df = pd.DataFrame(
        [
            {
                "prompt_id": p["prompt_id"],
                "layer": p["layer"],
                "sub_category": p["sub_category"],
                "difficulty": p.get("difficulty", "auto"),
            }
            for p in prompts
        ]
    )
    return judgments.merge(prompt_df, on="prompt_id", how="left")


def _covered_mask(df: pd.DataFrame) -> pd.Series:
    if "judge_error" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    err = df["judge_error"]
    return err.isna() | (err.astype(str).str.strip() == "")


# ---------------------------------------------------------------------------
# Per-dimension scoring — the key differentiator from T2I eval
# ---------------------------------------------------------------------------


def per_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model scoring broken down by evaluation dimension.

    Explodes per-atom answers and groups by the `dimension` tag on each atom
    to produce instruction_following_gm, visual_consistency_gm, etc.
    """
    rows = []
    for _, rec in df.iterrows():
        for a in rec.get("answers") or []:
            dim = a.get("dimension", "unknown")
            prob = a.get("probability")
            if prob is None:
                ans = str(a.get("answer", "")).strip().lower()
                prob = 1.0 if ans.startswith("y") else math.exp(DEFAULT_LOGPROB_FLOOR)
            rows.append(
                {
                    "model": rec["model"],
                    "dimension": dim,
                    "probability": float(prob),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["model"] + [f"{d}_am" for d in DIMENSIONS] + [f"{d}_gm" for d in DIMENSIONS]
        )
    qdf = pd.DataFrame(rows)

    result_rows = []
    for model, grp in qdf.groupby("model"):
        row: dict[str, Any] = {"model": model}
        for dim in DIMENSIONS:
            dim_probs = grp[grp["dimension"] == dim]["probability"].tolist()
            row[f"{dim}_am"] = round(soft_tifa_am(dim_probs), 4) if dim_probs else None
            row[f"{dim}_gm"] = round(soft_tifa_gm(dim_probs), 4) if dim_probs else None
        result_rows.append(row)

    out = pd.DataFrame(result_rows)
    sort_col = "instruction_following_gm"
    if sort_col in out.columns and out[sort_col].notna().any():
        out = out.sort_values(sort_col, ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standard aggregations (matching T2I eval patterns)
# ---------------------------------------------------------------------------


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    covered_df = df[_covered_mask(df)].copy()

    full = df.groupby("model", as_index=False).agg(
        overall_am=("score_am", "mean"),
        overall_gm=("score_gm", "mean"),
        std_dev_am=("score_am", "std"),
        std_dev_gm=("score_gm", "std"),
        n_total=("score_am", "count"),
    )

    covered = covered_df.groupby("model", as_index=False).agg(
        overall_am_covered=("score_am", "mean"),
        overall_gm_covered=("score_gm", "mean"),
        n_covered=("score_am", "count"),
    )

    out = full.merge(covered, on="model", how="left")
    out["n_covered"] = out["n_covered"].fillna(0).astype(int)
    out["n_total"] = out["n_total"].astype(int)
    out["coverage_rate"] = (out["n_covered"] / out["n_total"].clip(lower=1)).round(4)

    out["_sort_key"] = out["overall_gm_covered"].fillna(out["overall_gm"])
    out = (
        out.sort_values("_sort_key", ascending=False)
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    for col in [
        "overall_am",
        "overall_gm",
        "std_dev_am",
        "std_dev_gm",
        "overall_am_covered",
        "overall_gm_covered",
    ]:
        if col in out.columns:
            out[col] = out[col].round(4)

    return out


def per_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pivot_am = (
        df.groupby(["model", "sub_category"])["score_am"].mean().unstack("sub_category").round(4)
    )
    pivot_gm = (
        df.groupby(["model", "sub_category"])["score_gm"].mean().unstack("sub_category").round(4)
    )
    pivot_am.columns = [f"{c}__am" for c in pivot_am.columns]
    pivot_gm.columns = [f"{c}__gm" for c in pivot_gm.columns]
    pivot = pivot_am.join(pivot_gm, how="outer")
    pivot["overall_am"] = (
        pivot[[c for c in pivot.columns if c.endswith("__am")]].mean(axis=1).round(4)
    )
    pivot["overall_gm"] = (
        pivot[[c for c in pivot.columns if c.endswith("__gm")]].mean(axis=1).round(4)
    )
    return pivot.sort_values("overall_gm", ascending=False).reset_index()


def layer_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for metric, col_prefix in [("score_am", "am"), ("score_gm", "gm")]:
        piece = df.groupby(["model", "layer"])[metric].mean().unstack("layer").round(4)
        rename_map = {}
        for col in piece.columns:
            if col in (1, "1", "layer1_gold"):
                rename_map[col] = f"layer1_gold_{col_prefix}"
            elif col in (2, "2", "layer2_proprietary"):
                rename_map[col] = f"layer2_proprietary_{col_prefix}"
        piece = piece.rename(columns=rename_map)
        piece = piece[[c for c in piece.columns if c.startswith("layer")]].copy()
        l1 = f"layer1_gold_{col_prefix}"
        l2 = f"layer2_proprietary_{col_prefix}"
        if l1 in piece.columns and l2 in piece.columns:
            piece[f"divergence_{col_prefix}"] = (piece[l1] - piece[l2]).round(4)
        pieces.append(piece)
    out = pieces[0].join(pieces[1], how="outer") if len(pieces) == 2 else pieces[0]
    return out.reset_index()


def failure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in df.iterrows():
        for a in rec.get("answers") or []:
            rows.append(
                {
                    "model": rec["model"],
                    "sub_category": rec.get("sub_category"),
                    "q_type": a.get("type") or "unknown",
                    "dimension": a.get("dimension") or "unknown",
                    "answer": a.get("answer"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "q_type", "dimension", "failure_rate", "n"])
    qdf = pd.DataFrame(rows)
    qdf["is_fail"] = (qdf["answer"] == "no").astype(int)
    out = (
        qdf.groupby(["model", "q_type", "dimension"])
        .agg(failure_rate=("is_fail", "mean"), n=("is_fail", "count"))
        .round(4)
        .reset_index()
    )
    return out.sort_values(["model", "failure_rate"], ascending=[True, False])


def filter_rates(prompts: list[dict], judgments_df: pd.DataFrame) -> pd.DataFrame:
    n_total = len({p["prompt_id"] for p in prompts})
    if judgments_df.empty:
        return pd.DataFrame(
            columns=["model", "n_covered", "n_total", "uncovered", "uncovered_rate"]
        )

    covered = (
        judgments_df[_covered_mask(judgments_df)].groupby("model")["prompt_id"].nunique().to_dict()
    )

    models = judgments_df["model"].unique()
    rows = []
    for m in models:
        n_cov = covered.get(m, 0)
        rows.append(
            {
                "model": m,
                "n_covered": n_cov,
                "n_total": n_total,
                "uncovered": n_total - n_cov,
                "uncovered_rate": round((n_total - n_cov) / max(n_total, 1), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("uncovered_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_aggregation() -> dict[str, Path]:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompt_set()
    judgments = _load_all_judgments()
    if judgments.empty:
        log.error("No judgments to aggregate. Run the judge first.")
        return {}

    merged = _merge(judgments, prompts)

    paths: dict[str, Path] = {}

    lb = leaderboard(merged)
    lb.to_csv(SCORES_DIR / "leaderboard.csv", index=False)
    paths["leaderboard"] = SCORES_DIR / "leaderboard.csv"

    psc = per_subcategory(merged)
    psc.to_csv(SCORES_DIR / "per_subcategory.csv", index=False)
    paths["per_subcategory"] = SCORES_DIR / "per_subcategory.csv"

    pd_dim = per_dimension(merged)
    pd_dim.to_csv(SCORES_DIR / "per_dimension.csv", index=False)
    paths["per_dimension"] = SCORES_DIR / "per_dimension.csv"

    lc = layer_comparison(merged)
    lc.to_csv(SCORES_DIR / "layer_comparison.csv", index=False)
    paths["layer_comparison"] = SCORES_DIR / "layer_comparison.csv"

    fa = failure_analysis(merged)
    fa.to_csv(SCORES_DIR / "failure_analysis.csv", index=False)
    paths["failure_analysis"] = SCORES_DIR / "failure_analysis.csv"

    fr = filter_rates(prompts, merged)
    fr.to_csv(SCORES_DIR / "filter_rates.csv", index=False)
    paths["filter_rates"] = SCORES_DIR / "filter_rates.csv"

    summary = {
        "n_models": int(merged["model"].nunique()),
        "n_prompts_judged": int(merged["prompt_id"].nunique()),
        "n_total_judgments": int(len(merged)),
        "mean_score_am_overall": float(round(merged["score_am"].mean(), 4)),
        "mean_score_gm_overall": float(round(merged["score_gm"].mean(), 4)),
        "leaderboard_top3": lb.head(3).to_dict(orient="records"),
        "dimensions": pd_dim.to_dict(orient="records") if not pd_dim.empty else [],
    }
    with open(SCORES_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    paths["summary"] = SCORES_DIR / "summary_stats.json"

    log.info("Aggregation complete: %s", list(paths))
    return paths
