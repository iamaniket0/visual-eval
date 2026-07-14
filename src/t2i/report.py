"""Stage 5: Report Generation.

Produces:
    outputs/reports/aggregate_report.pdf    (lead magnet; all models)
    outputs/reports/{model_id}_card.pdf     (per-model pitch cards)

Charts are saved to outputs/reports/charts/ and embedded.
Every report carries the required T2I-CompBench++ disclosure.

Soft-TIFA migration: GM is the primary metric on every chart and ordering.
AM appears as a secondary / diagnostic view. The methodology and disclosure
sections adapt their text to the judge backend recorded in settings.yaml
(qwen_soft / gpt4o_soft / gpt4o_hard) so we cite the right paper + flag any
self-bias caveat when gpt-4o is judging gpt_image_15.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.core.utils import get_logger
from src.t2i import OUTPUTS_DIR, load_settings
from src.t2i.aggregator import worst_examples_for_model

log = get_logger("report")

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


# ---------------------------------------------------------------------------
# Judge-backend aware methodology text
# ---------------------------------------------------------------------------


def _current_judge_backend() -> str:
    try:
        return load_settings().get("judge", {}).get("backend", "gpt4o_hard")
    except Exception:
        return "gpt4o_hard"


def _methodology_text() -> str:
    """Methodology paragraph that matches the judge backend actually used."""
    backend = _current_judge_backend()
    base = (
        "Prompts are drawn from T2I-CompBench++ (Layer 1, 150 prompts) and a "
        "proprietary internally-authored set (Layer 2, 60 prompts). Each prompt is "
        "decomposed into atomic binary questions following the CompQuest "
        "pattern. "
    )
    if backend == "qwen_together_soft":
        # Actual backend used for the April 2026 run. Qwen3.5-397B-A17B on
        # Together AI serverless: open-source, preserves logprobs, no self-
        # preference bias against GPT Image family models.
        try:
            slug = load_settings().get("judge", {}).get("model_slug", "Qwen/Qwen3.5-397B-A17B")
        except Exception:
            slug = "Qwen/Qwen3.5-397B-A17B"
        judge = (
            f"Judge: {slug} (Qwen3.5 MoE, open-source) served on Together AI "
            "(serverless text+image endpoint). Scoring follows Soft-TIFA "
            "(Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec 2025): per-atom "
            'probabilities are extracted from the judge\'s "Yes" token logprob '
            "and aggregated two ways. <b>AM</b> = atom-level arithmetic mean of "
            "probabilities (partial-credit view, comparable to legacy TIFA). "
            "<b>GM</b> = prompt-level geometric mean (exp(mean(log p_i)), "
            "clipped at exp(-10)); GM is the primary metric because it collapses "
            "whenever any single atom is weak - the stricter view Kamath et al. "
            "show correlates best with human-labelled alignment (AUROC 94.5%). "
            "Thinking-mode is disabled on the judge request so Qwen3.5 emits a "
            'single-token "Yes"/"No" answer and logprob extraction is clean.'
        )
    elif backend == "qwen_soft":
        judge = (
            "Judge: Qwen3-VL (open-source) via OpenRouter. Scoring follows "
            "Soft-TIFA (Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec "
            "2025): per-atom probabilities are extracted from the judge's "
            '"Yes" token logprob and aggregated two ways. '
            "<b>AM</b> = atom-level arithmetic mean of probabilities "
            "(the partial-credit view, comparable to legacy TIFA). "
            "<b>GM</b> = prompt-level geometric mean "
            "(exp(mean(log p_i)), clipped at exp(-10)); GM is the "
            "primary metric here because it collapses whenever any single "
            "atom is weak - the stricter view Kamath et al. show correlates "
            "best with human-labelled alignment (AUROC 94.5%)."
        )
    elif backend == "gpt4o_soft":
        judge = (
            "Judge: GPT-4o via OpenRouter (temperature 0). Scoring follows "
            "Soft-TIFA (Kamath et al., GenEval 2, arXiv 2512.16853v1, Dec "
            "2025): per-atom probabilities are extracted from the judge's "
            '"Yes" token logprob. <b>AM</b> is the atom-level arithmetic '
            "mean (partial credit, comparable to legacy TIFA). <b>GM</b> is "
            "the prompt-level geometric mean, clipped at exp(-10) to avoid "
            "log(0), and is the primary metric here. "
            "Caveat: when GPT Image 1.5 (gpt_image_15) is in the benchmark, "
            "a known ~3-7 point self-preference bias inflates its judged "
            "score under this backend. The preferred open-source Qwen3-VL "
            "judge is blocked on provider logprob support as of this run; "
            "flip `judge.backend` to `qwen_soft` once that path opens."
        )
    else:  # gpt4o_hard or unknown
        judge = (
            "GPT-4o serves as the MLLM judge (temperature 0). Score per "
            "image = yes_count / total_questions (hard TIFA)."
        )
    tail = (
        " Human validation on 10% of images targets Cohen's kappa > 0.6. "
        f"Theme-level cuts apply an n&ge;{THEME_MIN_N} per-cell filter to "
        "the chart and top/bottom lists so statistically noisy themes "
        "don't dominate the narrative."
    )
    return base + judge + tail


def _disclosure_text() -> str:
    """Disclosure paragraph. Adds the Soft-TIFA comparability note when
    the judge backend is soft so readers know old + new runs aren't
    directly comparable."""
    backend = _current_judge_backend()
    parts = [DISCLOSURE_LAYERS]
    if backend in ("qwen_soft", "gpt4o_soft", "qwen_together_soft"):
        parts.append(
            "Scoring methodology: Soft-TIFA (Kamath et al., arXiv "
            "2512.16853v1, Dec 2025). Meta's paper reports "
            "Soft-TIFA-GM with Qwen3-VL at 94.5% AUROC on "
            "human-judged alignment versus 91.6% for legacy "
            "TIFA+GPT-4o. Previous runs of this benchmark used hard "
            "TIFA with GPT-4o and are not directly comparable to "
            "current results."
        )
    return "  ".join(parts)


def _pitch_backend_caveat() -> str:
    """A sentence to append to model-card data pitch when relevant."""
    if _current_judge_backend() == "gpt4o_soft":
        return (
            " Note: scores under GPT-4o as judge carry a documented "
            "self-preference bias when evaluating GPT Image family models; "
            "migration to an open-source Qwen judge is planned."
        )
    return ""


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


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
        ax.barh(range(len(models)), vals, color=GM_COLOR)
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
        gms_cov,
        height=0.28,
        color=GM_COLOR,
        label="GM on covered prompts (quality)",
    )
    ax.barh(
        [i + 0.25 for i in y],
        gms_full,
        height=0.28,
        facecolor="none",
        edgecolor="#1f3d66",
        hatch="///",
        linewidth=0.6,
        label="GM on full benchmark (quality x reliability)",
    )
    ax.barh(
        [i - 0.10 for i in y],
        ams_cov,
        height=0.22,
        color=AM_COLOR,
        label="AM on covered (diagnostic)",
    )
    ax.barh(
        [i - 0.10 for i in y],
        ams_full,
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


# ---------------------------------------------------------------------------
# PDF building blocks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def build_aggregate_report() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")
    psc = pd.read_csv(SCORES_DIR / "per_subcategory.csv")
    lc_path = SCORES_DIR / "layer_comparison.csv"
    lc = pd.read_csv(lc_path) if lc_path.exists() else pd.DataFrame()
    fr_path = SCORES_DIR / "filter_rates.csv"
    fr = pd.read_csv(fr_path) if fr_path.exists() else pd.DataFrame()

    lb_chart = chart_leaderboard(lb)
    sc_chart_gm = chart_subcategory(psc, metric="gm")
    sc_chart_am = chart_subcategory(psc, metric="am")
    lc_chart_gm = chart_layer_comparison(lc, metric="gm") if not lc.empty else None
    lc_chart_am = chart_layer_comparison(lc, metric="am") if not lc.empty else None

    out = REPORTS_DIR / "aggregate_report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    st = _styles()
    story = []

    story.append(Paragraph("T2I Compositional Faithfulness Benchmark", st["H1"]))
    story.append(
        Paragraph(
            "Where frontier text-to-image models still fail: numeracy, complex "
            "compositions, and 3D spatial reasoning.",
            st["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    # ------- Executive summary -------
    story.append(Paragraph("Executive Summary", st["H2"]))
    primary_score_col = "overall_gm" if "overall_gm" in lb.columns else "overall_score"
    ordered = lb.sort_values(primary_score_col, ascending=False).reset_index(drop=True)
    top = ordered.iloc[0]
    bottom = ordered.iloc[-1]

    # Weakness count under the primary metric. For Soft-TIFA we use GM
    # columns; legacy runs fall back to the old single score column.
    gm_cat_cols = [c for c in psc.columns if c.endswith("__gm")]
    if gm_cat_cols:
        value_cols = gm_cat_cols
        weakness_label = "GM"
    else:
        value_cols = [c for c in psc.columns if c not in ("model", "overall")]
        weakness_label = "score"
    weak_threshold = 0.85
    n_with_weakness = 0
    subcat_melt = []
    if value_cols:
        for _, r in psc.iterrows():
            scores = {c: r[c] for c in value_cols if pd.notna(r[c])}
            if not scores:
                continue
            if any(s < weak_threshold for s in scores.values()):
                n_with_weakness += 1
            for c, s in scores.items():
                subcat_melt.append((r["model"], c.replace("__gm", ""), float(s)))

    # Report the quality number on each model's covered set (apples-to-apples)
    # and call out any coverage shortfall separately so reviewers can see the
    # reliability discount without it collapsing into the quality claim.
    have_covered = (
        "overall_gm_covered" in lb.columns
        and "overall_am_covered" in lb.columns
        and "n_covered" in lb.columns
        and "n_total" in lb.columns
    )
    if have_covered:
        quality_col = "overall_gm_covered"
        ordered_q = (
            lb.assign(_q=lb[quality_col].fillna(lb["overall_gm"]))
            .sort_values("_q", ascending=False)
            .drop(columns="_q")
            .reset_index(drop=True)
        )
        top_q = ordered_q.iloc[0]
        bot_q = ordered_q.iloc[-1]
        top_q_val = float(
            top_q.get(quality_col) if pd.notna(top_q.get(quality_col)) else top_q["overall_gm"]
        )
        bot_q_val = float(
            bot_q.get(quality_col) if pd.notna(bot_q.get(quality_col)) else bot_q["overall_gm"]
        )
        n_total = int(lb["n_total"].iloc[0])
        summary_head = (
            f"Across {n_total} benchmark prompts and "
            f"{len(lb)} frontier T2I model{'s' if len(lb) != 1 else ''}, "
            f"compositional faithfulness on each model's covered prompts "
            f"(GM, strict, apples-to-apples) ranged from "
            f"<b>{top_q_val:.2f}</b> ({top_q['model']}) to "
            f"<b>{bot_q_val:.2f}</b> ({bot_q['model']}). "
        )
    else:
        summary_head = (
            f"Across {int(lb['n_prompts'].iloc[0])} prompts per model and "
            f"{len(lb)} frontier T2I model{'s' if len(lb) != 1 else ''}, "
            f"compositional faithfulness (GM, strict) ranged from "
            f"<b>{top[primary_score_col]:.2f}</b> ({top['model']}) to "
            f"<b>{bottom[primary_score_col]:.2f}</b> ({bottom['model']}). "
        )
    if n_with_weakness > 0:
        summary_tail = (
            f"{n_with_weakness} of {len(lb)} models show sub-category "
            f"{weakness_label} scores below {weak_threshold:.2f}, indicating "
            "substantial headroom for targeted training data."
        )
    elif subcat_melt:
        worst = min(subcat_melt, key=lambda t: t[2])
        best = max(subcat_melt, key=lambda t: t[2])
        summary_tail = (
            f"Models achieve strong overall scores with variance across "
            f"sub-categories, ranging from <b>{worst[2]:.2f}</b> ({worst[0]} "
            f"on {worst[1]}) to <b>{best[2]:.2f}</b> ({best[0]} on {best[1]})."
        )
    else:
        summary_tail = (
            "Per-sub-category scores were not available for this run; see "
            "the per-sub-category table below for details."
        )
    story.append(Paragraph(summary_head + summary_tail, st["Normal"]))
    story.append(Spacer(1, 10))

    # ------- Quality vs Reliability split (new, per review feedback) -------
    if have_covered:
        story.append(Paragraph("Quality vs. Reliability: Two Separate Stories", st["H2"]))
        lines = []
        for _, r in lb.sort_values(
            lb["overall_gm_covered"].fillna(lb["overall_gm"]).name, ascending=False
        ).iterrows():
            n_cov = int(r["n_covered"])
            n_tot = int(r["n_total"])
            uncov = n_tot - n_cov
            gm_cov = float(
                r["overall_gm_covered"] if pd.notna(r["overall_gm_covered"]) else r["overall_gm"]
            )
            am_cov = float(
                r["overall_am_covered"] if pd.notna(r["overall_am_covered"]) else r["overall_am"]
            )
            cov_pct = 100.0 * n_cov / max(1, n_tot)
            reliability_clause = (
                "(100% coverage)"
                if uncov == 0
                else f"(<b>{uncov}/{n_tot}</b> prompts uncovered &mdash; "
                f"{100.0 - cov_pct:.1f}% of benchmark)"
            )
            lines.append(
                f"<b>{r['model']}</b> &mdash; quality: AM "
                f"<b>{am_cov:.3f}</b> / GM <b>{gm_cov:.3f}</b> on "
                f"{n_cov} covered prompts. Reliability: {reliability_clause}."
            )
        for line in lines:
            story.append(Paragraph(line, st["Normal"]))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "<b>Quality</b> is measured on each model's covered prompts "
                "(apples-to-apples, same metric, excluding prompts the model "
                "produced no image for). <b>Reliability</b> is measured by the "
                "uncovered-rate &mdash; prompts where the model returned no image "
                "after retries. A full-benchmark GM score that blends the two "
                "(`GM (full)` column below) is shown for completeness but should "
                "not be quoted as a quality number in isolation.",
                st["Disclosure"],
            )
        )
        story.append(Spacer(1, 10))

    # ------- Overall ranking -------
    story.append(Paragraph("Overall Ranking", st["H2"]))
    story.append(
        Paragraph(
            "<i>Table columns: <b>AM/GM (covered)</b> = quality on the prompts "
            "a model actually produced images for (apples-to-apples, the "
            "number to quote). <b>AM/GM (full)</b> = same metric computed over "
            "all 210 prompts with uncovered prompts scored 0 (blends quality "
            "and reliability). <b>coverage</b> = n_covered / n_total. "
            "Sort order: GM (covered). GM = Soft-TIFA geometric mean, strict; "
            "AM = arithmetic mean, diagnostic.</i>",
            st["Disclosure"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(RLImage(str(lb_chart), width=6.5 * inch, height=4.1 * inch))
    story.append(Spacer(1, 6))
    story.append(_df_to_table(_leaderboard_display_df(lb)))
    story.append(PageBreak())

    # ------- Per-sub-category (GM primary, AM secondary) -------
    story.append(Paragraph("Per-Sub-Category Performance", st["H2"]))
    if sc_chart_gm:
        story.append(RLImage(str(sc_chart_gm), width=6.5 * inch, height=4.1 * inch))
        story.append(Spacer(1, 6))
    if sc_chart_am:
        story.append(
            Paragraph("<i>AM (diagnostic, shown below for comparison):</i>", st["Disclosure"])
        )
        story.append(RLImage(str(sc_chart_am), width=6.5 * inch, height=4.1 * inch))
        story.append(Spacer(1, 6))
    story.append(_df_to_table(psc))
    story.append(PageBreak())

    # ------- Theme breakdown -------
    tb_path = SCORES_DIR / "theme_breakdown.csv"
    if tb_path.exists():
        tb = pd.read_csv(tb_path)
        if not tb.empty:
            tb_filtered = tb[tb["n_prompts"] >= THEME_MIN_N]
            story.append(Paragraph("Performance by Theme", st["H2"]))
            story.append(
                Paragraph(
                    "Themes are multi-label tags extracted from prompt text "
                    "(a single prompt may carry 2-5 themes). This view slices "
                    "the same prompts along orthogonal axes - domain, setting, "
                    "composition density, attribute axes - surfacing where "
                    "scores diverge between models.",
                    st["Normal"],
                )
            )
            story.append(Spacer(1, 4))
            story.append(Paragraph(THEME_FILTER_NOTE, st["Disclosure"]))
            story.append(Spacer(1, 6))
            tv_chart = chart_theme_variance(tb, top_n=5)
            if tv_chart:
                story.append(RLImage(str(tv_chart), width=6.5 * inch, height=4.6 * inch))
                story.append(Spacer(1, 6))
            body_style = ParagraphStyle(
                name="ThemeCell", parent=_styles()["Normal"], fontSize=8, leading=10
            )
            score_col = "mean_score_gm" if "mean_score_gm" in tb.columns else "mean_score"
            header = "GM" if score_col == "mean_score_gm" else "score"
            rows = [
                [
                    Paragraph("<b>model</b>", body_style),
                    Paragraph(f"<b>top themes (by {header}, n&ge;{THEME_MIN_N})</b>", body_style),
                    Paragraph(
                        f"<b>bottom themes (by {header}, n&ge;{THEME_MIN_N})</b>", body_style
                    ),
                ]
            ]
            for m in sorted(tb_filtered["model"].unique()):
                m_tb = tb_filtered[tb_filtered["model"] == m].sort_values(
                    score_col, ascending=False
                )
                top_s = "<br/>".join(
                    f"{r.theme} &mdash; {getattr(r, score_col):.2f} (n={int(r.n_prompts)})"
                    for r in m_tb.head(5).itertuples()
                )
                bot_s = "<br/>".join(
                    f"{r.theme} &mdash; {getattr(r, score_col):.2f} (n={int(r.n_prompts)})"
                    for r in m_tb.tail(5)[::-1].itertuples()
                )
                rows.append(
                    [
                        Paragraph(m, body_style),
                        Paragraph(top_s or "(no qualifying themes)", body_style),
                        Paragraph(bot_s or "(no qualifying themes)", body_style),
                    ]
                )
            tv_tbl = Table(rows, hAlign="LEFT", colWidths=[1.6 * inch, 2.55 * inch, 2.55 * inch])
            tv_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(tv_tbl)
            story.append(PageBreak())

    # ------- Layer 1 vs Layer 2 (both AM and GM divergences) -------
    if lc_chart_gm:
        story.append(Paragraph("Layer 1 vs Layer 2: The Saturation Wedge", st["H2"]))
        story.append(
            Paragraph(
                "When public-benchmark (Layer 1) scores are high but proprietary "
                "(Layer 2) scores are materially lower, the gold benchmark has "
                "saturated and the proprietary prompts are still discriminating. "
                "This gap is where targeted training data moves the needle. "
                "<i>Expect GM divergence to differ from AM divergence - GM "
                "amplifies any prompt where a single atom collapses, so "
                "divergence signs can diverge between the two metrics.</i>",
                st["Normal"],
            )
        )
        story.append(Spacer(1, 6))
        story.append(RLImage(str(lc_chart_gm), width=6.5 * inch, height=4.1 * inch))
        story.append(Spacer(1, 6))
        if lc_chart_am:
            story.append(Paragraph("<i>AM view (diagnostic):</i>", st["Disclosure"]))
            story.append(RLImage(str(lc_chart_am), width=6.5 * inch, height=4.1 * inch))
            story.append(Spacer(1, 6))
        story.append(_df_to_table(lc))
        story.append(PageBreak())

    if not fr.empty:
        story.append(Paragraph("Reliability: Prompt Coverage", st["H2"]))
        story.append(
            Paragraph(
                "<b>uncovered_rate</b> is the fraction of the 210-benchmark where "
                "no image was ever produced, across all retries and any rescue "
                "runs. It is the best single reliability number from a customer "
                "perspective: a prompt that lands here would silently fail if "
                "submitted to the provider's API. Split into three diagnostic "
                "columns: <i>filtered</i> (explicit safety refusal), <i>errored</i> "
                "(API returned an error), and <i>uncovered</i> (no image ever "
                "returned for any seed after retries &mdash; can include silent "
                "empty-response cases such as intent-misrouting, which are NOT "
                "the same as a safety refusal and should be spot-checked on the "
                "specific prompts before attributing a cause).",
                st["Normal"],
            )
        )
        story.append(Spacer(1, 6))
        story.append(_df_to_table(fr))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Methodology", st["H2"]))
    story.append(Paragraph(_methodology_text(), st["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Disclosure", st["H2"]))
    story.append(Paragraph(_disclosure_text(), st["Disclosure"]))

    doc.build(story)
    log.info("Wrote aggregate report: %s", out)
    return out


# ---------------------------------------------------------------------------
# Per-model card
# ---------------------------------------------------------------------------


def build_model_card(model: str, pitch_text: str | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")
    psc = pd.read_csv(SCORES_DIR / "per_subcategory.csv")
    lc_path = SCORES_DIR / "layer_comparison.csv"
    lc = pd.read_csv(lc_path) if lc_path.exists() else pd.DataFrame()

    if model not in lb["model"].values:
        log.warning("Model %s not in leaderboard; skipping card", model)
        return None

    primary_score_col = "overall_gm" if "overall_gm" in lb.columns else "overall_score"
    secondary_score_col = "overall_am" if "overall_am" in lb.columns else None
    lb_sorted = lb.sort_values(primary_score_col, ascending=False).reset_index(drop=True)
    model_row = lb_sorted[lb_sorted["model"] == model].iloc[0]
    rank = int(lb_sorted.reset_index().index[lb_sorted["model"] == model][0]) + 1
    sub_chart = chart_model_subcategory(model, psc, metric="gm")
    worst = worst_examples_for_model(model, n=3)

    out = REPORTS_DIR / f"{model}_card.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    st = _styles()
    story = []

    story.append(Paragraph(f"T2I Benchmark Scorecard: {model}", st["H1"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Executive Summary", st["H2"]))
    score_str = f"<b>{model_row[primary_score_col]:.2f}</b> (GM)"
    if secondary_score_col:
        score_str += f" / <b>{model_row[secondary_score_col]:.2f}</b> (AM)"
    story.append(
        Paragraph(
            f"<b>{model}</b> scored {score_str} on compositional "
            f"faithfulness, ranking <b>#{rank}</b> of {len(lb)} frontier T2I "
            "models benchmarked. The scorecard below breaks down sub-category "
            "performance and highlights specific failure modes.",
            st["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    if sub_chart:
        story.append(Paragraph("Sub-Category Breakdown (GM)", st["H2"]))
        story.append(RLImage(str(sub_chart), width=5.5 * inch, height=3.2 * inch))
        story.append(Spacer(1, 10))

    # ------- Thematic strengths/weaknesses (GM primary) -------
    tb_path = SCORES_DIR / "theme_breakdown.csv"
    if tb_path.exists():
        tb_all = pd.read_csv(tb_path)
        score_col = "mean_score_gm" if "mean_score_gm" in tb_all.columns else "mean_score"
        m_tb_full = tb_all[tb_all["model"] == model].sort_values(score_col, ascending=False)
        m_tb = m_tb_full[m_tb_full["n_prompts"] >= THEME_MIN_N]
        if not m_tb.empty:
            story.append(Paragraph("Thematic Strengths and Weaknesses (GM)", st["H2"]))
            story.append(
                Paragraph(
                    "Themes are multi-label tags (a prompt may carry 2-5 themes), "
                    "so these scores slice the same images along orthogonal axes "
                    "to the sub-category chart above. Strongest themes show where "
                    "this model is comfortable; weakest themes are the most "
                    "concrete candidates for targeted training data.",
                    st["Normal"],
                )
            )
            story.append(Spacer(1, 4))
            story.append(Paragraph(THEME_FILTER_NOTE, st["Disclosure"]))
            story.append(Spacer(1, 6))
            n_qualifying = len(m_tb)
            top3 = m_tb.head(min(3, n_qualifying))
            bot3 = m_tb.tail(min(3, n_qualifying)).iloc[::-1]
            if n_qualifying < 3:
                story.append(
                    Paragraph(
                        f"<i>Only {n_qualifying} theme{'s' if n_qualifying != 1 else ''} "
                        f"pass the n&ge;{THEME_MIN_N} threshold for this model; all "
                        "shown below. Themes below the threshold excluded for "
                        "statistical robustness.</i>",
                        st["Normal"],
                    )
                )
                story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Strongest themes</b>", st["Normal"]))
            for r in top3.itertuples():
                story.append(
                    Paragraph(
                        f"- <b>{r.theme}</b>: {getattr(r, score_col):.2f} "
                        f"(n={int(r.n_prompts)}) &mdash; this model's "
                        "best-performing theme, well ahead of its overall score.",
                        st["Normal"],
                    )
                )
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Weakest themes</b>", st["Normal"]))
            for r in bot3.itertuples():
                story.append(
                    Paragraph(
                        f"- <b>{r.theme}</b>: {getattr(r, score_col):.2f} "
                        f"(n={int(r.n_prompts)}) &mdash; candidate for targeted "
                        "training data; see failure examples below.",
                        st["Normal"],
                    )
                )
            story.append(Spacer(1, 10))

    # ------- Divergence narrative (read GM divergence when available) -------
    div = None
    if not lc.empty and model in lc["model"].values:
        row = lc[lc["model"] == model].iloc[0]
        story.append(Paragraph("Public Benchmark vs Proprietary Prompts", st["H2"]))
        # Prefer GM view for the headline narrative; fall back to legacy.
        if "layer1_gold_gm" in row.index:
            l1 = row.get("layer1_gold_gm")
            l2 = row.get("layer2_proprietary_gm")
            div = row.get("divergence_gm")
            metric_label = "GM"
        else:
            l1 = row.get("layer1_gold")
            l2 = row.get("layer2_proprietary")
            div = row.get("divergence")
            metric_label = ""
        if pd.notna(l1) and pd.notna(l2):
            if pd.notna(div) and div > 0.1:
                layer_narrative = (
                    "Strong positive divergence suggests the public benchmark "
                    "is saturated for this model; proprietary prompts are "
                    "where the remaining weakness lives."
                )
            elif pd.notna(div) and div < -0.1:
                layer_narrative = (
                    f"Layer 2 scored higher than Layer 1 by {abs(div):.2f}. "
                    "In this MVP run, that likely reflects the Layer 2 "
                    "starter set being easier on average than "
                    "T2I-CompBench++. A harder Layer 2 pass is planned."
                )
            else:
                layer_narrative = "Scores are consistent across layers."
            story.append(
                Paragraph(
                    f"Layer 1 (T2I-CompBench++): <b>{l1:.2f}</b>. "
                    f"Layer 2 (proprietary): <b>{l2:.2f}</b>. "
                    f"Divergence ({metric_label}): <b>{div:+.2f}</b>. " + layer_narrative,
                    st["Normal"],
                )
            )
        story.append(Spacer(1, 10))

    # ------- Failure examples with per-atom probabilities -------
    if worst:
        story.append(Paragraph("Failure Examples", st["H2"]))
        story.append(
            Paragraph(
                "Each atomic question shows the judge's probability that the "
                "image satisfies the constraint. ✓ = probability >= 0.50, "
                "✗ = probability < 0.50. GM (the primary prompt-level score) "
                "collapses whenever any single atom is weak, which is why a "
                "prompt can have 4 of 5 atoms passing and still land in the "
                "bottom of the distribution.",
                st["Normal"],
            )
        )
        story.append(Spacer(1, 8))
        for w in worst:
            gm = w.get("score_gm")
            am = w.get("score_am", w.get("score"))
            story.append(
                Paragraph(
                    f"<b>{w['prompt_id']}</b> ({w['sub_category']}) &mdash; "
                    f"GM={gm:.2f}, AM={am:.2f}"
                    if gm is not None
                    else f"<b>{w['prompt_id']}</b> ({w['sub_category']}) &mdash; score={w['score']:.2f}",
                    st["Normal"],
                )
            )
            story.append(Paragraph(f'Prompt: "{w["prompt_text"]}"', st["Normal"]))
            try:
                if w["image_path"] and Path(w["image_path"]).exists():
                    story.append(RLImage(w["image_path"], width=3 * inch, height=3 * inch))
            except Exception as e:
                log.warning("Could not embed image %s: %s", w["image_path"], e)
            # Sort atoms by probability descending so the pattern is visible:
            # clearly-passing atoms on top, then borderline, then misses.
            atoms = sorted(
                (w.get("answers") or []), key=lambda a: a.get("probability", 0.0), reverse=True
            )
            if atoms:
                lines = []
                for a in atoms[:7]:
                    p = a.get("probability")
                    mark = (
                        "&#10003;"
                        if (p is not None and p >= 0.5)
                        else (
                            "&#10007;"
                            if p is not None
                            else ("&#10003;" if a.get("answer") == "yes" else "&#10007;")
                        )
                    )
                    p_str = f"p={p:.2f}" if p is not None else "hard verdict"
                    q = a.get("question", "")
                    lines.append(f"{mark} [{p_str}] {q}")
                story.append(Paragraph("<br/>".join(lines), st["Normal"]))
            story.append(Spacer(1, 10))

    # ------- Training data recommendations (divergence-aware) -------
    story.append(PageBreak())
    story.append(Paragraph("Training Data Recommendations", st["H2"]))
    if pitch_text is not None:
        pitch = pitch_text
    elif div is not None and pd.notna(div) and div > 0.1:
        pitch = (
            "Targeted compositional training data — numeracy-rich "
            "scenes, multi-constraint prompt-image pairs, and 3D-spatial "
            "annotations — calibrated to the specific failure modes shown "
            "above can close the gap between Layer 1 "
            "and Layer 2 scores within one training cycle."
        )
    elif div is not None and pd.notna(div) and div < -0.1:
        pitch = (
            "This model's scores on the proprietary Layer 2 prompts exceed "
            "its scores on the public T2I-CompBench++ benchmark, indicating "
            "the current Layer 2 sample is under-calibrated relative to the "
            "public set. A harder Layer 2 pass with higher object counts and "
            "more constraints is planned. The per-sub-category failure modes "
            "shown above remain actionable signal for training data targeting."
        )
    else:
        pitch = (
            "Targeted compositional training data calibrated to "
            "the specific failure modes shown above can address the "
            "weaknesses identified. While this model shows "
            "consistent performance across public and proprietary prompt "
            "sets, the per-sub-category failure analysis identifies concrete "
            "areas for improvement."
        )
    pitch += _pitch_backend_caveat()
    story.append(Paragraph(pitch, st["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Methodology", st["H2"]))
    story.append(Paragraph(_methodology_text(), st["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Disclosure", st["H2"]))
    story.append(Paragraph(_disclosure_text(), st["Disclosure"]))

    doc.build(story)
    log.info("Wrote model card: %s", out)
    return out


def build_all_reports() -> list[Path]:
    out = [build_aggregate_report()]
    lb = pd.read_csv(SCORES_DIR / "leaderboard.csv")
    for model in lb["model"]:
        card = build_model_card(model)
        if card:
            out.append(card)
    return out
