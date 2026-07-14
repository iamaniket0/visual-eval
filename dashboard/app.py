"""Interactive results dashboard for Visual Eval.

Launch:
    streamlit run dashboard/app.py
    # or via CLI:
    visual-eval dashboard
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
T2I_SCORES = ROOT / "outputs" / "t2i" / "scores"
EDIT_SCORES = ROOT / "outputs" / "edit" / "scores"

st.set_page_config(
    page_title="Visual Eval Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def render_leaderboard(df: pd.DataFrame, title: str, gm_col: str, am_col: str):
    df_sorted = df.sort_values(gm_col, ascending=False).reset_index(drop=True)
    df_sorted.index += 1
    df_sorted.index.name = "Rank"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["model"][::-1],
        x=df_sorted[gm_col][::-1],
        orientation="h",
        name="GM (primary)",
        marker_color="#4a90e2",
        text=[f"{v:.3f}" for v in df_sorted[gm_col][::-1]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=df_sorted["model"][::-1],
        x=df_sorted[am_col][::-1],
        orientation="h",
        name="AM (diagnostic)",
        marker_color="#b0c9e4",
        text=[f"{v:.3f}" for v in df_sorted[am_col][::-1]],
        textposition="outside",
        visible="legendonly",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Score",
        xaxis_range=[0, 1.05],
        height=max(400, len(df_sorted) * 50 + 100),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    display_cols = ["model", gm_col, am_col]
    if "n_covered" in df_sorted.columns:
        display_cols.append("n_covered")
    if "n_total" in df_sorted.columns:
        display_cols.append("n_total")
    available = [c for c in display_cols if c in df_sorted.columns]
    st.dataframe(
        df_sorted[available].style.format(
            {c: "{:.3f}" for c in available if c in [gm_col, am_col]}
        ),
        use_container_width=True,
    )


def render_subcategory(df: pd.DataFrame, metric: str = "gm"):
    suffix = f"__{metric}"
    cat_cols = [c for c in df.columns if c.endswith(suffix)]
    if not cat_cols:
        cat_cols = [c for c in df.columns if c not in ("model", "overall")]
        suffix = ""
    if not cat_cols:
        st.info("No sub-category data available.")
        return

    melted = []
    for _, row in df.iterrows():
        for c in cat_cols:
            melted.append({
                "model": row["model"],
                "category": c.replace(suffix, ""),
                "score": row[c],
            })
    melted_df = pd.DataFrame(melted)

    fig = px.bar(
        melted_df, x="category", y="score", color="model",
        barmode="group", range_y=[0, 1],
        title=f"Per-Category Performance ({metric.upper()})",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        xaxis_title="", yaxis_title=f"Score ({metric.upper()})",
        height=500, margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_radar(df: pd.DataFrame, metric: str = "gm"):
    suffix = f"__{metric}"
    cat_cols = [c for c in df.columns if c.endswith(suffix)]
    if not cat_cols:
        return
    categories = [c.replace(suffix, "") for c in cat_cols]

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[c] for c in cat_cols] + [row[cat_cols[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row["model"],
            line_color=colors[i % len(colors)],
            fill="toself",
            opacity=0.3,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Model Comparison Radar ({metric.upper()})",
        height=500,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dimension_heatmap(df: pd.DataFrame):
    dim_cols = [c for c in df.columns if c not in ("model",)]
    if not dim_cols:
        return
    fig = px.imshow(
        df.set_index("model")[dim_cols],
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=1,
        title="Edit Evaluation: Per-Dimension Heatmap",
        aspect="auto",
    )
    fig.update_layout(height=max(300, len(df) * 40 + 100))
    st.plotly_chart(fig, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────

st.title("Visual Eval Dashboard")
st.markdown("Interactive results viewer for T2I generation and image editing benchmarks.")

tab_t2i, tab_edit, tab_compare = st.tabs(["T2I Evaluation", "Edit Evaluation", "Compare"])

# ─── T2I Tab ──────────────────────────────────────────────────
with tab_t2i:
    lb = load_csv(T2I_SCORES / "leaderboard.csv")
    psc = load_csv(T2I_SCORES / "per_subcategory.csv")
    lc = load_csv(T2I_SCORES / "layer_comparison.csv")
    tb = load_csv(T2I_SCORES / "theme_breakdown.csv")

    if lb is not None:
        st.header("T2I Leaderboard")
        gm_col = "overall_gm_covered" if "overall_gm_covered" in lb.columns else "overall_gm"
        am_col = "overall_am_covered" if "overall_am_covered" in lb.columns else "overall_am"
        if gm_col not in lb.columns:
            gm_col = "overall_score"
            am_col = "overall_score"
        render_leaderboard(lb, "T2I Model Ranking (Soft-TIFA)", gm_col, am_col)

        if psc is not None:
            st.header("Sub-Category Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                render_subcategory(psc, "gm")
            with col2:
                render_radar(psc, "gm")

        if lc is not None:
            st.header("Layer 1 vs Layer 2 Comparison")
            l1_col = "layer1_gold_gm" if "layer1_gold_gm" in lc.columns else "layer1_gold"
            l2_col = "layer2_proprietary_gm" if "layer2_proprietary_gm" in lc.columns else "layer2_proprietary"
            if l1_col in lc.columns and l2_col in lc.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=lc["model"], y=lc[l1_col],
                    name="Layer 1 (Public Benchmark)", marker_color="#4a90e2",
                ))
                fig.add_trace(go.Bar(
                    x=lc["model"], y=lc[l2_col],
                    name="Layer 2 (Proprietary)", marker_color="#e27d4a",
                ))
                fig.update_layout(
                    barmode="group", yaxis_range=[0, 1],
                    title="Public vs Proprietary Prompt Performance",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

        if tb is not None and not tb.empty:
            st.header("Theme Analysis")
            score_col = "mean_score_gm" if "mean_score_gm" in tb.columns else "mean_score"
            selected_model = st.selectbox(
                "Select model", sorted(tb["model"].unique()), key="t2i_theme_model"
            )
            m_tb = tb[tb["model"] == selected_model].sort_values(score_col)
            fig = px.bar(
                m_tb, x=score_col, y="theme", orientation="h",
                title=f"Theme Scores for {selected_model}",
                range_x=[0, 1], color=score_col,
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(height=max(400, len(m_tb) * 25 + 100), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No T2I results found. Run the pipeline first:\n\n"
            "```bash\n"
            "visual-eval t2i generate --models sanity\n"
            "visual-eval t2i judge\n"
            "visual-eval t2i aggregate\n"
            "```"
        )

# ─── Edit Tab ─────────────────────────────────────────────────
with tab_edit:
    edit_lb = load_csv(EDIT_SCORES / "leaderboard.csv")
    edit_dim = load_csv(EDIT_SCORES / "per_dimension.csv")

    if edit_lb is not None:
        st.header("Edit Model Leaderboard")
        gm_col = "overall_gm" if "overall_gm" in edit_lb.columns else "overall_score"
        am_col = "overall_am" if "overall_am" in edit_lb.columns else gm_col
        render_leaderboard(edit_lb, "Edit Model Ranking (Soft-TIFA)", gm_col, am_col)

        if edit_dim is not None:
            st.header("Per-Dimension Scores")
            render_dimension_heatmap(edit_dim)
    else:
        st.info(
            "No edit results found. Run the pipeline first:\n\n"
            "```bash\n"
            "visual-eval edit run --models sanity\n"
            "visual-eval edit judge\n"
            "visual-eval edit aggregate\n"
            "```"
        )

# ─── Compare Tab ──────────────────────────────────────────────
with tab_compare:
    st.header("Cross-Pipeline Comparison")
    st.markdown(
        "Compare scoring distributions between T2I generation and image editing models. "
        "Note: scores are not directly comparable across pipelines — T2I uses single-image "
        "judging while edit uses dual-image (source + edited) judging."
    )

    t2i_lb = load_csv(T2I_SCORES / "leaderboard.csv")
    edit_lb2 = load_csv(EDIT_SCORES / "leaderboard.csv")

    if t2i_lb is not None and edit_lb2 is not None:
        t2i_gm = "overall_gm" if "overall_gm" in t2i_lb.columns else "overall_score"
        edit_gm = "overall_gm" if "overall_gm" in edit_lb2.columns else "overall_score"

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=t2i_lb[t2i_gm], name="T2I Models",
            marker_color="#4a90e2", boxpoints="all",
            text=t2i_lb["model"],
        ))
        fig.add_trace(go.Box(
            y=edit_lb2[edit_gm], name="Edit Models",
            marker_color="#e27d4a", boxpoints="all",
            text=edit_lb2["model"],
        ))
        fig.update_layout(
            title="Score Distribution: T2I vs Edit Models",
            yaxis_title="GM Score", yaxis_range=[0, 1],
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("T2I Models", len(t2i_lb))
            st.metric("T2I Mean GM", f"{t2i_lb[t2i_gm].mean():.3f}")
            st.metric("T2I Best", t2i_lb.loc[t2i_lb[t2i_gm].idxmax(), "model"])
        with col2:
            st.metric("Edit Models", len(edit_lb2))
            st.metric("Edit Mean GM", f"{edit_lb2[edit_gm].mean():.3f}")
            st.metric("Edit Best", edit_lb2.loc[edit_lb2[edit_gm].idxmax(), "model"])
    else:
        st.info("Run both T2I and edit pipelines to see cross-pipeline comparisons.")
