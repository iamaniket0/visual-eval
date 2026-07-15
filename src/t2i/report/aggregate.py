"""Aggregate report builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
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
from src.t2i.report.charts import (
    chart_layer_comparison,
    chart_leaderboard,
    chart_subcategory,
    chart_theme_variance,
)
from src.t2i.report.constants import (
    REPORTS_DIR,
    SCORES_DIR,
    THEME_FILTER_NOTE,
    THEME_MIN_N,
)
from src.t2i.report.methodology import _disclosure_text, _methodology_text
from src.t2i.report.pdf_helpers import _df_to_table, _leaderboard_display_df, _styles

log = get_logger("report")


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
        _top_raw = top_q.get(quality_col) if pd.notna(top_q.get(quality_col)) else top_q["overall_gm"]
        top_q_val = float(_top_raw)  # type: ignore[arg-type]
        _bot_raw = bot_q.get(quality_col) if pd.notna(bot_q.get(quality_col)) else bot_q["overall_gm"]
        bot_q_val = float(_bot_raw)  # type: ignore[arg-type]
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
            str(lb["overall_gm_covered"].fillna(lb["overall_gm"]).name), ascending=False
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
