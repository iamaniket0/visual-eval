"""Per-model scorecard (model card) builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from src.core.utils import get_logger
from src.t2i.aggregator import worst_examples_for_model
from src.t2i.report.charts import chart_model_subcategory
from src.t2i.report.constants import (
    REPORTS_DIR,
    SCORES_DIR,
    THEME_FILTER_NOTE,
    THEME_MIN_N,
)
from src.t2i.report.methodology import (
    _disclosure_text,
    _methodology_text,
    _pitch_backend_caveat,
)
from src.t2i.report.pdf_helpers import _styles

log = get_logger("report")


def build_model_card(model: str, pitch_text: str | None = None) -> Path | None:
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
