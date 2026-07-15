"""Chart generation for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.t2i.report.constants import AM_COLOR, CHARTS_DIR, GM_COLOR, THEME_MIN_N


def _save_fig(fig, name: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_leaderboard(lb: pd.DataFrame) -> Path:
    """Horizontal grouped bars with the quality/reliability split baked in.

    Two bars per model:
      * Covered GM  -- apples-to-apples quality number (prompts the model
        actually produced). This is the bar the eye should land on.
      * Full GM     -- same metric but with missing-image prompts scored 0.
        Shown as a lighter overlay so the reader can see the reliability
        discount in the same frame.

    AM variants (thinner, muted) sit alongside for diagnostic comparison.
    Sort order: covered GM descending so quality leads. When covered and
    full are identical (100% coverage), the two bars collapse visually
    which is exactly the right signal.
    """
    if not ("overall_gm" in lb.columns and "overall_am" in lb.columns):
        # Legacy single-score schema: degenerate to the original bar.
        ordered = lb.sort_values("overall_score", ascending=False).reset_index(drop=True)
        vals = ordered["overall_score"][::-1].values
        models = ordered["model"][::-1].values
        fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(models) + 1.5))
        ax.barh(range(len(models)), list(vals), color=GM_COLOR)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Compositional faithfulness (legacy)")
        return _save_fig(fig, "leaderboard.png")

    # Primary sort: covered GM (falls back to full GM where no coverage col).
    sort_key = (
        lb["overall_gm_covered"].fillna(lb["overall_gm"])
        if "overall_gm_covered" in lb.columns
        else lb["overall_gm"]
    )
    ordered = (
        lb.assign(_sort=sort_key)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )

    gms_full = ordered["overall_gm"][::-1].values
    ams_full = ordered["overall_am"][::-1].values
    gms_cov = (
        ordered["overall_gm_covered"].fillna(ordered["overall_gm"])[::-1].values
        if "overall_gm_covered" in ordered.columns
        else gms_full
    )
    ams_cov = (
        ordered["overall_am_covered"].fillna(ordered["overall_am"])[::-1].values
        if "overall_am_covered" in ordered.columns
        else ams_full
    )
    coverage = (
        (
            ordered["n_covered"].astype(int).astype(str)
            + "/"
            + ordered["n_total"].astype(int).astype(str)
        )[::-1].values
        if "n_covered" in ordered.columns and "n_total" in ordered.columns
        else [""] * len(ordered)
    )
    models = ordered["model"][::-1].values
    y = list(range(len(models)))

    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(models) + 2.0))
    # Covered = primary (solid). Full = reliability-discounted (hatched overlay).
    ax.barh(
        [i + 0.25 for i in y],
        list(gms_cov),
        height=0.28,
        color=GM_COLOR,
        label="GM on covered prompts (quality)",
    )
    ax.barh(
        [i + 0.25 for i in y],
        list(gms_full),
        height=0.28,
        facecolor="none",
        edgecolor="#1f3d66",
        hatch="///",
        linewidth=0.6,
        label="GM on full benchmark (quality x reliability)",
    )
    ax.barh(
        [i - 0.10 for i in y],
        list(ams_cov),
        height=0.22,
        color=AM_COLOR,
        label="AM on covered (diagnostic)",
    )
    ax.barh(
        [i - 0.10 for i in y],
        list(ams_full),
        height=0.22,
        facecolor="none",
        edgecolor="#6b4a1f",
        hatch="///",
        linewidth=0.6,
        label="AM on full (diagnostic)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}\n({c})" for m, c in zip(models, coverage)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Compositional faithfulness (Soft-TIFA)")
    ax.set_title("Overall Ranking  -  covered-only (quality) vs full-set (quality x reliability)")

    for i, (gm_c, gm_f, am_c, am_f) in enumerate(zip(gms_cov, gms_full, ams_cov, ams_full)):
        ax.text(gm_c + 0.01, i + 0.25, f"{gm_c:.2f}", va="center", fontsize=8)
        ax.text(am_c + 0.01, i - 0.10, f"{am_c:.2f}", va="center", fontsize=8, color="#555555")
        # Only annotate full bars when they differ materially from covered
        # (>0.005) so 100%-coverage rows don't get visual noise.
        if abs(gm_c - gm_f) > 0.005:
            ax.text(gm_f - 0.05, i + 0.25, f"{gm_f:.2f}", va="center", fontsize=7, color="#1f3d66")
    ax.legend(loc="lower right", fontsize=7)
    return _save_fig(fig, "leaderboard.png")


def chart_subcategory(psc: pd.DataFrame, metric: str = "gm") -> Path | None:
    """Per-(model, sub_category) bar chart. `metric` in {"gm", "am"}.

    With the Soft-TIFA schema, the CSV has columns like `numeracy__am` +
    `numeracy__gm`. Select the suffix that matches `metric`. Legacy CSVs
    (pre-migration) have bare sub-category names; we fall back to those
    so old reports still render.
    """
    suffix = f"__{metric}"
    cat_cols = [c for c in psc.columns if c.endswith(suffix)]
    if cat_cols:
        display_cols = {c: c.replace(suffix, "") for c in cat_cols}
    else:
        # Legacy schema: pre-migration CSVs.
        display_cols = {c: c for c in psc.columns if c not in ("model", "overall")}
        if not display_cols:
            return None
    sub = psc[["model"] + list(display_cols.keys())].copy()
    sub = sub.rename(columns=display_cols)
    value_cols = list(display_cols.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    sub.set_index("model")[value_cols].plot(kind="bar", ax=ax)
    ax.set_ylabel(f"Mean score ({metric.upper()})")
    ax.set_ylim(0, 1)
    label = "GM (primary)" if metric == "gm" else "AM (diagnostic)"
    ax.set_title(f"Per-Sub-Category Performance - {label}")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    return _save_fig(fig, f"per_subcategory_{metric}.png")


def chart_layer_comparison(lc: pd.DataFrame, metric: str = "gm") -> Path | None:
    """Layer 1 vs Layer 2 bars per model. `metric` in {"gm", "am"}."""
    l1 = f"layer1_gold_{metric}"
    l2 = f"layer2_proprietary_{metric}"
    if l1 not in lc.columns or l2 not in lc.columns:
        # Legacy bare columns?
        l1 = "layer1_gold"
        l2 = "layer2_proprietary"
        if l1 not in lc.columns or l2 not in lc.columns:
            return None
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(lc))
    ax.bar(
        [i - 0.2 for i in x], lc[l1], width=0.4, label="Layer 1 (T2I-CompBench++)", color=GM_COLOR
    )
    ax.bar([i + 0.2 for i in x], lc[l2], width=0.4, label="Layer 2 (proprietary)", color="#e27d4a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(lc["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Mean score ({metric.upper()})")
    label = "GM (primary)" if metric == "gm" else "AM (diagnostic)"
    ax.set_title(f"Public Benchmark vs Proprietary Prompts - {label}")
    ax.legend()
    return _save_fig(fig, f"layer_comparison_{metric}.png")


def chart_theme_variance(tb: pd.DataFrame, top_n: int = 5, min_n: int = THEME_MIN_N) -> Path | None:
    """Horizontal grouped bars: top-n themes with the highest across-model
    spread under GM (the primary metric)."""
    if tb.empty:
        return None
    score_col = "mean_score_gm" if "mean_score_gm" in tb.columns else "mean_score"
    filtered = tb[tb["n_prompts"] >= min_n]
    if filtered.empty:
        return None
    counts = filtered.groupby("theme")["model"].nunique()
    full_coverage = counts[counts == tb["model"].nunique()].index
    filtered = filtered[filtered["theme"].isin(full_coverage)]
    if filtered.empty:
        return None
    pivot = filtered.pivot(index="theme", columns="model", values=score_col)
    if pivot.shape[1] < 2:
        return None
    spread = pivot.std(axis=1).sort_values(ascending=False)
    top_themes = spread.head(top_n).index.tolist()
    if not top_themes:
        return None
    sub = pivot.loc[top_themes]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(top_themes) * sub.shape[1] + 2))
    sub.plot(kind="barh", ax=ax, width=0.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    label = "GM" if score_col == "mean_score_gm" else "mean score"
    ax.set_xlabel(f"{label} within theme")
    ax.set_title(f"Most Discriminating Themes (top {top_n} by across-model spread, {label})")
    ax.legend(title="model", fontsize=8, loc="lower right")
    return _save_fig(fig, "theme_variance.png")


def chart_model_subcategory(model: str, psc: pd.DataFrame, metric: str = "gm") -> Path | None:
    """Per-model sub-category bar chart under the chosen metric."""
    row = psc[psc["model"] == model]
    if row.empty:
        return None
    suffix = f"__{metric}"
    cat_cols = [c for c in psc.columns if c.endswith(suffix)]
    if cat_cols:
        display_cols = [c.replace(suffix, "") for c in cat_cols]
        values = row[cat_cols].values.flatten()
    else:
        # Legacy schema.
        value_cols = [c for c in psc.columns if c not in ("model", "overall")]
        display_cols = value_cols
        values = row[value_cols].values.flatten()
        metric = "score"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(display_cols, values, color=GM_COLOR)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Score ({metric.upper()})")
    ax.set_title(f"{model} - Sub-Category Breakdown ({metric.upper()})")
    plt.xticks(rotation=20, ha="right")
    return _save_fig(fig, f"{model}_subcategory_{metric}.png")
