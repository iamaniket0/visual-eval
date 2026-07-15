"""Stage 4: Score aggregation.

Reads outputs/judgments/{model}.jsonl + prompts/prompt_set.json and produces:
    outputs/scores/leaderboard.csv         (AM + GM + std + seed std)
    outputs/scores/per_subcategory.csv     (AM + GM per sub-category)
    outputs/scores/layer_comparison.csv    (AM + GM divergence per layer)
    outputs/scores/failure_analysis.csv    (per q_type failure rate)
    outputs/scores/theme_breakdown.csv     (AM + GM per theme, multi-label)
    outputs/scores/filter_rates.csv
    outputs/scores/summary_stats.json

Soft-TIFA migration:
    - Each judgment record carries `score_am` + `score_gm` in addition to
      the legacy `score` field (Kamath et al., GenEval 2, 2025).
    - AM = mean of per-atom probabilities (partial-credit view).
    - GM = exp(mean(log(p_i))), clipped to exp(logprob_floor) to avoid
      log(0). GM is the STRICT view that collapses whenever any single atom
      is weak - it's the primary metric in the report.
    - Legacy records without the new fields: we infer probability per atom
      from `answer` (1.0 for "yes", 0.0 for "no", floored to exp(-10)), then
      compute AM/GM from those. Old single-score runs become readable
      under the new aggregator without re-judging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.bootstrap import _gm_stat, bootstrap_ci
from src.core.scoring import (
    probabilities_from_answers,
    soft_tifa_am,
    soft_tifa_gm,
)
from src.core.utils import get_logger, read_jsonl
from src.t2i import OUTPUTS_DIR, PROMPTS_DIR
from src.t2i.prompt_loader import load_prompt_set

log = get_logger("aggregator")

SCORES_DIR = OUTPUTS_DIR / "scores"

# Local aliases for backward compatibility.
_probabilities_from_answers = probabilities_from_answers


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_prompt_themes() -> dict[str, list[str]]:
    """Load the optional prompt -> themes mapping."""
    path = PROMPTS_DIR / "prompt_themes.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("prompt_themes.json unreadable (%s); skipping theme breakdown", e)
        return {}
    if not isinstance(data, dict):
        log.warning("prompt_themes.json has unexpected shape; skipping theme breakdown")
        return {}
    return data


def _load_all_judgments() -> pd.DataFrame:
    """Flatten judgments from all models into a single DataFrame.

    Each row == one (model, prompt_id, seed) triple == one judgment record.
    Carries both score_am and score_gm; legacy records without those fields
    have them computed here via `_probabilities_from_answers` so the rest
    of the aggregator is agnostic to which judge backend produced the data.
    """
    rows = []
    for path in (OUTPUTS_DIR / "judgments").glob("*.jsonl"):
        model = path.stem
        for rec in read_jsonl(path):
            answers = rec.get("answers", []) or []
            # Preference order for the per-prompt score_am / score_gm:
            #   1. Fields the new judge backends write explicitly.
            #   2. The legacy single `score` field (hard TIFA) - if no new
            #      fields are present we treat the hard score as both AM
            #      AND GM for legacy-record back-compat. This is the right
            #      choice for files written before the migration: the old
            #      single number preserved both rankings the rest of the
            #      aggregator needs.
            #   3. Fall back to deriving probabilities from per-atom
            #      answer strings only if neither is present.
            score_am = rec.get("score_am")
            score_gm = rec.get("score_gm")
            legacy_score = rec.get("score")
            if score_am is None and score_gm is None and legacy_score is not None:
                score_am = float(legacy_score)
                score_gm = float(legacy_score)
            elif score_am is None or score_gm is None:
                probs = _probabilities_from_answers(answers)
                score_am = soft_tifa_am(probs) if score_am is None else float(score_am)
                score_gm = soft_tifa_gm(probs) if score_gm is None else float(score_gm)
            rows.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "model": model,
                    "seed": int(rec.get("seed") or 0),
                    "score": legacy_score if legacy_score is not None else score_am,
                    "score_am": float(score_am),
                    "score_gm": float(score_gm),
                    "judge_error": rec.get("error"),
                    "answers": answers,
                }
            )
    if not rows:
        log.warning("No judgments found")
        return pd.DataFrame(
            columns=[
                "prompt_id",
                "model",
                "seed",
                "score",
                "score_am",
                "score_gm",
                "judge_error",
                "answers",
            ]
        )
    return pd.DataFrame(rows)


def _collapse_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-seed judgment rows to one row per (model, prompt_id).

    Both AM and GM are averaged across seeds, yielding the per-prompt
    scores that feed every downstream aggregation. `seed_std_dev_am` /
    `seed_std_dev_gm` measure how much the score bounces across seeds -
    the statistical-defensibility signal.
    """
    if df.empty:
        return df

    def _pick_answers(group):
        group = group.sort_values("seed")
        for a in group["answers"]:
            if a:
                return a
        return []

    # Split the frame into "good" seeds (no judge error) and "errored" seeds.
    # Good seeds drive score_am/score_gm averaging so partial-coverage prompts
    # (e.g. seed 0 failed, seeds 1-2 scored) report the quality of the seeds
    # that actually produced images instead of a diluted average including
    # the zero from the missing seed.
    #
    # `judge_error` on the collapsed row is kept NON-NULL only when every
    # seed for this (model, prompt_id) errored - a prompt where at least one
    # seed got scored counts as covered.
    err_mask = df["judge_error"].notna() & (df["judge_error"].astype(str).str.strip() != "")
    good = df[~err_mask]
    bad = df[err_mask]

    if not good.empty:
        good_agg = good.groupby(["model", "prompt_id"], as_index=False).agg(
            score_am=("score_am", "mean"),
            score_gm=("score_gm", "mean"),
            seed_std_dev_am=("score_am", "std"),
            seed_std_dev_gm=("score_gm", "std"),
            n_seeds=("seed", "count"),
        )
    else:
        good_agg = pd.DataFrame(
            columns=[
                "model",
                "prompt_id",
                "score_am",
                "score_gm",
                "seed_std_dev_am",
                "seed_std_dev_gm",
                "n_seeds",
            ]
        )

    # Every (model, prompt_id) that appears in the raw frame must appear in
    # the collapsed output, even if every seed errored: downstream aggregators
    # (failure_analysis, filter_rates) need the row to show up so coverage
    # accounting is correct. Reconstruct the full index from `df` and
    # left-join the good-seed scores; rows missing from `good_agg` represent
    # fully-uncovered prompts and get score 0 + judge_error set.
    full_idx = df[["model", "prompt_id"]].drop_duplicates()
    agg = full_idx.merge(good_agg, on=["model", "prompt_id"], how="left")

    if not bad.empty:
        bad_err = bad.groupby(["model", "prompt_id"])["judge_error"].first().reset_index()
        agg = agg.merge(bad_err, on=["model", "prompt_id"], how="left")
    else:
        agg["judge_error"] = None

    # For (model, prompt_id) pairs where at least one good seed scored, the
    # quality columns are populated and judge_error stays NaN (covered). For
    # pairs where every seed errored, score columns are NaN -> fill with 0
    # and keep the error string (uncovered).
    fully_uncovered = agg["score_am"].isna()
    agg.loc[fully_uncovered, "score_am"] = 0.0
    agg.loc[fully_uncovered, "score_gm"] = 0.0
    agg.loc[~fully_uncovered, "judge_error"] = None

    for col in ["seed_std_dev_am", "seed_std_dev_gm"]:
        agg[col] = agg[col].fillna(0.0)
    agg["n_seeds"] = agg["n_seeds"].fillna(0).astype(int)

    # Legacy `score` column stays around == score_am for back-compat with
    # any consumer that hasn't yet learned about AM/GM.
    agg["score"] = agg["score_am"]
    # Back-compat alias for callers that read `seed_std_dev` (single field).
    agg["seed_std_dev"] = agg["seed_std_dev_am"]

    answers_rows = []
    for (m, pid), grp in df.groupby(["model", "prompt_id"]):
        answers_rows.append({"model": m, "prompt_id": pid, "answers": _pick_answers(grp)})
    answers_col = pd.DataFrame(answers_rows)
    agg = agg.merge(answers_col, on=["model", "prompt_id"], how="left")
    return agg


def _load_generation_log() -> pd.DataFrame:
    recs = read_jsonl(OUTPUTS_DIR / "metadata" / "generation_log.jsonl")
    if not recs:
        return pd.DataFrame(columns=["prompt_id", "model", "status"])
    return pd.DataFrame(recs)


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


# ---------------------------------------------------------------------------
# Downstream aggregations (all emit AM + GM columns)
# ---------------------------------------------------------------------------


def _covered_mask(df: pd.DataFrame) -> pd.Series:
    """Per-row boolean: True if the prompt was actually scored by the judge.

    A prompt is NOT covered when either the generator produced no image for
    any seed, OR every seed's judgment came back with a hard error (no atoms
    scored). Those prompts sit in the frame with score_am=score_gm=0 and
    `judge_error` populated. The covered-only view excludes them so the
    leaderboard shows quality on the intersection of prompts every model
    actually answered -- the apples-to-apples comparison. The full-set view
    (which DOES keep them as zeros) still travels alongside for the
    reliability story.
    """
    if "judge_error" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    err = df["judge_error"]
    return err.isna() | (err.astype(str).str.strip() == "")


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Overall ranking under Soft-TIFA.

    Emits two parallel views per model:
      * `overall_am` / `overall_gm`  -- full set (missing/filtered prompts
        stay in as zeros; this is the reliability-blended number).
      * `overall_am_covered` / `overall_gm_covered` -- apples-to-apples
        score restricted to prompts the model actually produced and got
        judged on.
    Coverage columns: `n_covered`, `n_total`, `coverage_rate`.

    Sort order is by `overall_gm_covered` so the quality story leads; the
    full-set number is still available right next to it so reviewers can
    see the reliability discount explicitly. GM stays the primary strict
    metric throughout; AM is kept for diagnostic comparison.
    """
    covered_df = df[_covered_mask(df)].copy()

    full_kwargs: dict[str, Any] = {
        "overall_am": ("score_am", "mean"),
        "overall_gm": ("score_gm", "mean"),
        "std_dev_am": ("score_am", "std"),
        "std_dev_gm": ("score_gm", "std"),
        "n_prompts": ("score_am", "count"),
    }
    if "seed_std_dev_am" in df.columns:
        full_kwargs["seed_std_dev_am"] = ("seed_std_dev_am", "mean")
        full_kwargs["seed_std_dev_gm"] = ("seed_std_dev_gm", "mean")
    full = df.groupby("model", as_index=False).agg(**full_kwargs)

    cov_kwargs: dict[str, Any] = {
        "overall_am_covered": ("score_am", "mean"),
        "overall_gm_covered": ("score_gm", "mean"),
        "n_covered": ("score_am", "count"),
    }
    covered = covered_df.groupby("model", as_index=False).agg(**cov_kwargs)

    out = full.merge(covered, on="model", how="left")
    # Models with zero covered prompts shouldn't NaN-out the quality
    # columns -- leave them empty and let the report handle the edge case.
    out["n_covered"] = out["n_covered"].fillna(0).astype(int)
    out = out.rename(columns={"n_prompts": "n_total"})
    out["n_total"] = out["n_total"].astype(int)
    out["coverage_rate"] = (out["n_covered"] / out["n_total"].clip(lower=1)).round(4)

    # Primary sort: covered GM (falls back to full GM for 0-coverage rows).
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
        "seed_std_dev_am",
        "seed_std_dev_gm",
        "overall_am_covered",
        "overall_gm_covered",
    ]:
        if col in out.columns:
            out[col] = out[col].round(4)

    ci_rows = []
    for model_name in out["model"]:
        model_scores = covered_df.loc[covered_df["model"] == model_name]
        am_vals = model_scores["score_am"].tolist()
        gm_vals = model_scores["score_gm"].tolist()
        am_lo, am_hi = bootstrap_ci(am_vals, stat_fn=np.mean)  # type: ignore[arg-type]
        gm_lo, gm_hi = bootstrap_ci(gm_vals, stat_fn=_gm_stat)
        ci_rows.append(
            {
                "model": model_name,
                "am_ci_lower": round(am_lo, 4),
                "am_ci_upper": round(am_hi, 4),
                "gm_ci_lower": round(gm_lo, 4),
                "gm_ci_upper": round(gm_hi, 4),
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    out = out.merge(ci_df, on="model", how="left")

    # Back-compat aliases - old consumers look for these names. `n_prompts`
    # stays the full-set count (210) for existing report code that reads it;
    # the new views read `n_total` / `n_covered` explicitly.
    out["n_prompts"] = out["n_total"]
    out["overall_score"] = out["overall_am"]
    out["std_dev"] = out["std_dev_am"]
    if "seed_std_dev_am" in out.columns:
        out["seed_std_dev"] = out["seed_std_dev_am"]
    return out


def per_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, sub_category) mean scores. Emits both AM and GM columns,
    suffixed `__am` / `__gm` for each sub-category, plus overall columns.
    """
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
    """Layer 1 vs Layer 2 scores. Divergence = sales wedge.

    Emits both AM and GM divergences. Expect these to tell slightly
    different stories - AM divergence is what we reported pre-migration;
    GM divergence is stricter and usually shows wider gaps.
    Layer 3 (hard-mode prompts) is reported separately if present.
    """
    if df.empty:
        return pd.DataFrame()
    # Only layers 1 and 2 participate in the divergence calculation.
    # Layer 3 is a different difficulty tier and comparing it to L1/L2
    # would confuse the "saturation wedge" narrative.
    df_l12 = df[df["layer"].isin([1, 2, "1", "2", "layer1_gold", "layer2_proprietary"])]
    if df_l12.empty:
        df_l12 = df
    pieces: list[pd.DataFrame] = []
    for metric, col_prefix in [("score_am", "am"), ("score_gm", "gm")]:
        piece = df_l12.groupby(["model", "layer"])[metric].mean().unstack("layer").round(4)
        rename_map = {}
        for col in piece.columns:
            if col in (1, "1", "layer1_gold"):
                rename_map[col] = f"layer1_gold_{col_prefix}"
            elif col in (2, "2", "layer2_proprietary"):
                rename_map[col] = f"layer2_proprietary_{col_prefix}"
        piece = piece.rename(columns=rename_map)
        # Drop any leftover layer columns (e.g., layer 3 if it slipped through)
        piece = piece[[c for c in piece.columns if str(c).startswith("layer")]].copy()
        l1 = f"layer1_gold_{col_prefix}"
        l2 = f"layer2_proprietary_{col_prefix}"
        if l1 in piece.columns and l2 in piece.columns:
            piece[f"divergence_{col_prefix}"] = (piece[l1] - piece[l2]).round(4)
        pieces.append(piece)
    out = pieces[0].join(pieces[1], how="outer")

    # Back-compat: the old schema had bare `layer1_gold` / `layer2_proprietary`
    # / `divergence` columns. Alias those to the AM variants so old consumers
    # keep working without edits.
    alias_map = {
        "layer1_gold_am": "layer1_gold",
        "layer2_proprietary_am": "layer2_proprietary",
        "divergence_am": "divergence",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    return out.reset_index()


def failure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question-type failure rate per model.

    Explodes the raw `answers` list so per-atom hard verdicts drive the
    failure rate. Soft-TIFA's per-atom `probability` isn't used here -
    this metric measures "how often is the atom below 0.5" which is the
    hard-verdict derived from probability.
    """
    rows = []
    for _, rec in df.iterrows():
        for a in rec.get("answers") or []:
            rows.append(
                {
                    "model": rec["model"],
                    "sub_category": rec.get("sub_category"),
                    "q_type": a.get("type") or "unknown",
                    "answer": a.get("answer"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "q_type", "failure_rate", "n"])
    qdf = pd.DataFrame(rows)
    qdf["is_fail"] = (qdf["answer"] == "no").astype(int)
    out = (
        qdf.groupby(["model", "q_type"])
        .agg(failure_rate=("is_fail", "mean"), n=("is_fail", "count"))
        .round(4)
        .reset_index()
    )
    return out.sort_values(["model", "failure_rate"], ascending=[True, False])


def theme_breakdown(df: pd.DataFrame, prompt_themes: dict[str, list[str]]) -> pd.DataFrame:
    """Per-model, per-theme mean AM and GM with std + count.

    Columns: model, theme,
             mean_score_am, std_dev_am,
             mean_score_gm, std_dev_gm,
             n_prompts,
             mean_score, std_dev    (back-compat aliases == AM).
    """
    cols = [
        "model",
        "theme",
        "mean_score_am",
        "std_dev_am",
        "mean_score_gm",
        "std_dev_gm",
        "n_prompts",
        "mean_score",
        "std_dev",
    ]
    if not prompt_themes or df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, rec in df.iterrows():
        for theme in prompt_themes.get(rec["prompt_id"], []) or []:
            rows.append(
                {
                    "model": rec["model"],
                    "theme": theme,
                    "score_am": float(rec["score_am"]),
                    "score_gm": float(rec["score_gm"]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=cols)

    qdf = pd.DataFrame(rows)
    out = (
        qdf.groupby(["model", "theme"])
        .agg(
            mean_score_am=("score_am", "mean"),
            std_dev_am=("score_am", "std"),
            mean_score_gm=("score_gm", "mean"),
            std_dev_gm=("score_gm", "std"),
            n_prompts=("score_am", "count"),
        )
        .reset_index()
    )
    for c in ["mean_score_am", "std_dev_am", "mean_score_gm", "std_dev_gm"]:
        out[c] = out[c].round(4)
    out["n_prompts"] = out["n_prompts"].astype(int)
    # Back-compat aliases.
    out["mean_score"] = out["mean_score_am"]
    out["std_dev"] = out["std_dev_am"]
    return out.sort_values(["model", "mean_score_gm"], ascending=[True, False]).reset_index(
        drop=True
    )


def filter_rates(
    gen_df: pd.DataFrame,
    covered_by_model: dict[str, int] | None = None,
    total_prompts: int | None = None,
) -> pd.DataFrame:
    """Per-model reliability table.

    Separates three failure modes a customer would see if they called the
    API naively on the same benchmark prompts:
      * `filtered`     -- provider returned an explicit safety refusal.
      * `errored`      -- provider returned an error (transient or hard).
      * `uncovered`    -- no image on disk after all retries/rescues; for a
                          customer this looks identical to "the API silently
                          failed on this prompt" (includes intent-misrouting
                          and permanent empty-response cases).
    `uncovered_rate` is the best single reliability number: fraction of the
    210-prompt benchmark where NO image was ever produced. Does not double-
    count: a prompt that eventually got an image on a retry is covered.
    """
    if gen_df.empty and not covered_by_model:
        return pd.DataFrame(
            columns=[
                "model",
                "filtered",
                "errored",
                "n_covered",
                "n_total",
                "uncovered",
                "uncovered_rate",
                "filter_rate",
            ]
        )
    if not gen_df.empty:
        summary = (
            gen_df.assign(
                filtered=(gen_df["status"] == "FILTERED").astype(int),
                errored=(gen_df["status"] == "ERROR").astype(int),
            )
            .groupby("model")[["filtered", "errored"]]
            .sum()
            .reset_index()
        )
    else:
        summary = pd.DataFrame({"model": list(covered_by_model or {})})
        summary["filtered"] = 0
        summary["errored"] = 0

    if covered_by_model is not None and total_prompts is not None:
        summary["n_covered"] = summary["model"].map(covered_by_model).fillna(0).astype(int)
        summary["n_total"] = int(total_prompts)
        summary["uncovered"] = (summary["n_total"] - summary["n_covered"]).clip(lower=0)
        summary["uncovered_rate"] = (summary["uncovered"] / summary["n_total"].clip(lower=1)).round(
            4
        )
    else:
        summary["n_covered"] = 0
        summary["n_total"] = 0
        summary["uncovered"] = 0
        summary["uncovered_rate"] = 0.0

    # Keep legacy `filter_rate` for back-compat with existing report code.
    # Uses `n_total` when available to avoid the misleading attempts-based
    # denominator (attempts include multi-seed and retries).
    denom = summary["n_total"].where(
        summary["n_total"] > 0, summary[["filtered", "errored"]].sum(axis=1).clip(lower=1)
    )
    summary["filter_rate"] = (summary["filtered"] / denom).round(4)
    return summary.sort_values("uncovered_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_aggregation() -> dict[str, Path]:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompt_set()
    judgments_raw = _load_all_judgments()
    if judgments_raw.empty:
        log.error("No judgments to aggregate. Run the judge first.")
        return {}
    # Collapse per-seed judgments to per-prompt scores for the main views;
    # keep the raw per-seed frame for failure_analysis which wants every
    # yes/no across all seeds to compute failure rates fairly.
    judgments = _collapse_seeds(judgments_raw)
    merged = _merge(judgments, prompts)
    merged_raw = _merge(judgments_raw, prompts)

    paths: dict[str, Path] = {}

    lb = leaderboard(merged)
    lb.to_csv(SCORES_DIR / "leaderboard.csv", index=False)
    paths["leaderboard"] = SCORES_DIR / "leaderboard.csv"

    psc = per_subcategory(merged)
    psc.to_csv(SCORES_DIR / "per_subcategory.csv", index=False)
    paths["per_subcategory"] = SCORES_DIR / "per_subcategory.csv"

    lc = layer_comparison(merged)
    lc.to_csv(SCORES_DIR / "layer_comparison.csv", index=False)
    paths["layer_comparison"] = SCORES_DIR / "layer_comparison.csv"

    fa = failure_analysis(merged_raw)
    fa.to_csv(SCORES_DIR / "failure_analysis.csv", index=False)
    paths["failure_analysis"] = SCORES_DIR / "failure_analysis.csv"

    # Coverage map per model comes from the covered view of the same judgments
    # that feed the leaderboard, so the reliability row in filter_rates always
    # agrees with the leaderboard's n_covered/n_total columns.
    covered_counts = merged[_covered_mask(merged)].groupby("model")["prompt_id"].nunique().to_dict()
    n_prompts_total = len({p["prompt_id"] for p in prompts})
    fr = filter_rates(
        _load_generation_log(), covered_by_model=covered_counts, total_prompts=n_prompts_total
    )
    fr.to_csv(SCORES_DIR / "filter_rates.csv", index=False)
    paths["filter_rates"] = SCORES_DIR / "filter_rates.csv"

    prompt_themes = _load_prompt_themes()
    if prompt_themes:
        tb = theme_breakdown(merged, prompt_themes)
        if not tb.empty:
            tb.to_csv(SCORES_DIR / "theme_breakdown.csv", index=False)
            paths["theme_breakdown"] = SCORES_DIR / "theme_breakdown.csv"

    summary = {
        "n_models": int(merged["model"].nunique()),
        "n_prompts_judged": int(merged["prompt_id"].nunique()),
        "n_total_judgments": int(len(merged)),
        "mean_score_am_overall": float(round(merged["score_am"].mean(), 4)),
        "mean_score_gm_overall": float(round(merged["score_gm"].mean(), 4)),
        "leaderboard_top3": lb.head(3).to_dict(orient="records"),
        # Wedge candidates remain based on GM (the stricter metric).
        "layer_wedge_candidates_gm": [
            row
            for row in lc.to_dict(orient="records")
            if row.get("divergence_gm") is not None and row["divergence_gm"] > 0.1
        ],
    }
    with open(SCORES_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    paths["summary"] = SCORES_DIR / "summary_stats.json"

    log.info("Aggregation complete: %s", list(paths))
    return paths


def worst_examples_for_model(model: str, n: int = 5) -> list[dict]:
    """Return the n worst-scoring (by GM) generations for the report."""
    prompts_by_id = {p["prompt_id"]: p for p in load_prompt_set()}
    judgments = read_jsonl(OUTPUTS_DIR / "judgments" / f"{model}.jsonl")
    judgments = [j for j in judgments if j.get("image_path")]

    # GM is the stricter metric - rank on it so failure examples line up
    # with the primary metric shown in the rest of the report.
    def _gm_score(j):
        gm = j.get("score_gm")
        if gm is None:
            gm = soft_tifa_gm(_probabilities_from_answers(j.get("answers", [])))
        return gm

    judgments.sort(key=_gm_score)
    out = []
    for j in judgments[:n]:
        p = prompts_by_id.get(j["prompt_id"], {})
        out.append(
            {
                "prompt_id": j["prompt_id"],
                "prompt_text": p.get("prompt_text", ""),
                "sub_category": p.get("sub_category", ""),
                "image_path": j.get("image_path"),
                "score": j.get("score", 0.0),
                "score_am": j.get("score_am", j.get("score", 0.0)),
                "score_gm": j.get("score_gm", _gm_score(j)),
                "answers": j.get("answers", []),
            }
        )
    return out
