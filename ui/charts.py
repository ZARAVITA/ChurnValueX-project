"""
ui/charts.py — All Plotly figures, theme-aware.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.theme import SEG_COLORS_LIGHT, SEG_COLORS_DARK

_LEGEND_BASE = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")


def _is_dark(t: dict) -> bool:
    """Detect dark mode from background color."""
    bg = t.get("bg_main", "#ffffff")
    # Dark mode backgrounds start with #0 or are very dark
    return bg.startswith("#0") or bg.startswith("#1")


def _layout(t: dict, legend_extra=None) -> dict:
    """Base layout dict. legend_extra dict is merged into the legend."""
    legend = dict(**_LEGEND_BASE)
    if legend_extra:
        legend.update(legend_extra)

    dark = _is_dark(t)
    if dark:
        plot_bg  = "rgba(255,255,255,0.04)"   # voile blanc très léger — distinct du fond page
        grid_col = "rgba(255,255,255,0.08)"
        axis_col = "rgba(255,255,255,0.12)"
        font_col = "#CBD5E1"                  # texte axes plus lisible
        legend.setdefault("bgcolor",      "rgba(13,20,32,0.82)")
        legend.setdefault("bordercolor",  "rgba(255,255,255,0.12)")
        legend.setdefault("borderwidth",  1)
        legend["font"] = dict(color="#F1F5F9", size=12)
    else:
        plot_bg  = t["plotly_plot"]
        grid_col = t["plotly_grid"]
        axis_col = t["plotly_grid"]
        font_col = t["plotly_font"]

    return dict(
        paper_bgcolor=t["plotly_paper"],
        plot_bgcolor =plot_bg,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=font_col, size=12),
        xaxis=dict(
            gridcolor=grid_col, zerolinecolor=grid_col, linecolor=axis_col,
            tickfont=dict(color=font_col), title_font=dict(color=font_col),
        ),
        yaxis=dict(
            gridcolor=grid_col, zerolinecolor=grid_col, linecolor=axis_col,
            tickfont=dict(color=font_col), title_font=dict(color=font_col),
        ),
        legend=legend,
        margin=dict(l=10, r=10, t=35, b=10),
    )


def _seg_colors(t: dict) -> dict:
    return SEG_COLORS_DARK if _is_dark(t) else SEG_COLORS_LIGHT


# ── Decision Matrix ───────────────────────────────────────────────────────────

def fig_decision_matrix(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    dark = _is_dark(t)

    # Fond légèrement différent du bg_main en mode sombre
    if dark:
        plot_bg   = "rgba(255,255,255,0.04)"   # voile blanc très léger
        paper_bg  = "rgba(255,255,255,0.0)"
        grid_col  = "rgba(255,255,255,0.08)"
        axis_col  = "rgba(255,255,255,0.12)"
        font_col  = "#CBD5E1"                  # texte axes + légende plus clair
        # Légende avec fond semi-transparent pour lisibilité
        legend_bg = "rgba(13, 20, 32, 0.82)"
        legend_bd = "rgba(255,255,255,0.12)"
    else:
        plot_bg   = "rgba(255,255,255,0)"
        paper_bg  = "rgba(255,255,255,0)"
        grid_col  = t["plotly_grid"]
        axis_col  = t["plotly_grid"]
        font_col  = t["plotly_font"]
        legend_bg = "rgba(255,255,255,0.92)"
        legend_bd = "rgba(0,0,0,0.08)"

    fig = px.scatter(
        df, x="CLV_12mois", y="Churn_proba",
        color="Matrix_Segment",
        color_discrete_map=sc,
        hover_data={"CustomerID": True, "CLV_12mois": ":.1f",
                    "Churn_proba": ":.1%", "Matrix_Segment": True},
        labels={"CLV_12mois": "Customer Lifetime Value (12M)", "Churn_proba": "Churn Risk"},
        opacity=0.75,
    )

    clv_med = df["CLV_12mois"].median()
    fig.add_hline(y=0.5,     line=dict(color="rgba(148,163,184,0.35)", dash="dot", width=1.5))
    fig.add_vline(x=clv_med, line=dict(color="rgba(148,163,184,0.35)", dash="dot", width=1.5))

    fig.update_traces(marker=dict(size=7, line=dict(width=0)))

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor =plot_bg,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=font_col, size=12),
        xaxis=dict(
            gridcolor=grid_col,
            zerolinecolor=grid_col,
            linecolor=axis_col,
            tickfont=dict(color=font_col),
            title_font=dict(color=font_col),
        ),
        yaxis=dict(
            gridcolor=grid_col,
            zerolinecolor=grid_col,
            linecolor=axis_col,
            tickfont=dict(color=font_col),
            title_font=dict(color=font_col),
            tickformat=".0%",
        ),
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=legend_bd,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color="#F1F5F9" if dark else "#0F172A",
                size=12,
            ),
            title=dict(
                text="Segment  ",
                font=dict(color="#94A3B8" if dark else "#475569", size=11),
            ),
            itemsizing="constant",
        ),
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Quadrant labels semi-transparents
    clv_max = df["CLV_12mois"].quantile(0.98)
    label_color = "rgba(203,213,225,0.45)" if dark else "rgba(71,85,105,0.3)"
    label_size  = 11

    for (x_pos, y_pos, txt) in [
        (clv_med * 0.3,  0.75, "OPTIMIZE"),
        (clv_med * 0.3,  0.25, "AUTOMATE"),
        (clv_med + (clv_max - clv_med) * 0.4, 0.75, "PRIORITY"),
        (clv_med + (clv_max - clv_med) * 0.4, 0.25, "PROTECT"),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos,
            text=txt,
            showarrow=False,
            font=dict(size=label_size, color=label_color,
                      family="DM Mono, monospace"),
            opacity=0.6,
        )

    return fig


# ── Before / After demo ───────────────────────────────────────────────────────

def fig_demo_before(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.scatter(df, x="CLV_12mois", y="Churn_proba",
                     color_discrete_sequence=["#94A3B8"],
                     labels={"CLV_12mois": "CLV", "Churn_proba": "Churn Risk"})
    fig.update_traces(marker=dict(size=6, opacity=0.4))
    fig.update_layout(
        **_layout(t),
        height=320,
        title=dict(text="Raw data — no structure", font=dict(size=12)),
        yaxis_tickformat=".0%",
    )
    return fig


def fig_demo_after(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    fig = px.scatter(df, x="CLV_12mois", y="Churn_proba",
                     color="Matrix_Segment", color_discrete_map=sc,
                     labels={"CLV_12mois": "CLV", "Churn_proba": "Churn Risk"})
    clv_med = df["CLV_12mois"].median()
    fig.add_hline(y=0.5,     line=dict(color="rgba(128,128,128,0.3)", dash="dot"))
    fig.add_vline(x=clv_med, line=dict(color="rgba(128,128,128,0.3)", dash="dot"))
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        **_layout(t, legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)),
        height=320,
        title=dict(text="4 action zones identified", font=dict(size=12)),
        yaxis_tickformat=".0%",
    )
    return fig


# ── Distributions ─────────────────────────────────────────────────────────────

def fig_clv_hist(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.histogram(df, x="CLV_12mois", nbins=50, color_discrete_sequence=[t["accent_blue"]])
    fig.update_layout(
        **_layout(t),
        height=270,
        xaxis_title="CLV (12 months)", yaxis_title="Customers", showlegend=False,
    )
    return fig


def fig_risk_hist(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.histogram(df, x="Churn_proba", nbins=50,
                       color_discrete_sequence=["#F59E0B"])
    fig.update_layout(
        **_layout(t),
        height=270,
        xaxis_title="Churn Probability", yaxis_title="Customers",
        showlegend=False, xaxis_tickformat=".0%",
    )
    return fig


def fig_seg_pie(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    dark = _is_dark(t)
    counts = df["Matrix_Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.pie(counts, names="Segment", values="Count",
                 color="Segment", color_discrete_map=sc, hole=0.55)
    fig.update_layout(
        **_layout(t, legend_extra=dict(orientation="h", yanchor="top", y=-0.05)),
        height=280,
    )
    # Surcharge séparée — paper_bgcolor est déjà dans _layout(), on l'écrase ici
    if dark:
        fig.update_layout(paper_bgcolor="rgba(255,255,255,0.04)")
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(color="#F1F5F9" if dark else "#0F172A"),
    )
    return fig


def fig_tenure_churn(df: pd.DataFrame, t: dict) -> go.Figure:
    if "Tenure" not in df.columns:
        return go.Figure()
    bins = pd.cut(df["Tenure"], bins=10)
    grp  = df.groupby(bins, observed=False)["Churn_proba"].mean().reset_index()
    grp["Tenure"] = grp["Tenure"].astype(str)
    fig = px.bar(grp, x="Tenure", y="Churn_proba",
                 color_discrete_sequence=[t["accent_blue"]])
    fig.update_layout(
        **_layout(t),
        height=270,
        xaxis_title="Tenure (months)", yaxis_title="Avg Churn Risk",
        xaxis_tickangle=-30, showlegend=False, yaxis_tickformat=".0%",
    )
    return fig


def fig_satisfaction_churn(df: pd.DataFrame, t: dict) -> go.Figure:
    if "SatisfactionScore" not in df.columns:
        return go.Figure()
    grp = df.groupby("SatisfactionScore")["Churn_proba"].mean().reset_index()
    fig = px.bar(grp, x="SatisfactionScore", y="Churn_proba",
                 color_discrete_sequence=["#F59E0B"])
    fig.update_layout(
        **_layout(t),
        height=270,
        xaxis_title="Satisfaction Score (1–5)", yaxis_title="Avg Churn Risk",
        showlegend=False, yaxis_tickformat=".0%",
    )
    return fig


def fig_clv_by_segment(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    grp = df.groupby("Matrix_Segment")["CLV_12mois"].median().reset_index()
    fig = px.bar(grp, x="Matrix_Segment", y="CLV_12mois",
                 color="Matrix_Segment", color_discrete_map=sc)
    fig.update_layout(
        **_layout(t),
        height=270,
        xaxis_title="Segment", yaxis_title="Median CLV ($)", showlegend=False,
    )
    return fig


def fig_survival_curves(df: pd.DataFrame, t: dict) -> go.Figure:
    if "Tenure" not in df.columns or "proba_churn_now" not in df.columns:
        return go.Figure()
    df2 = df.copy()
    df2["survival"] = 1 - df2["proba_churn_now"]
    bins = pd.cut(df2["Tenure"], bins=12)
    grp  = df2.groupby(bins, observed=False)["survival"].mean().reset_index()
    grp["Tenure"] = grp["Tenure"].astype(str)
    fig = px.line(grp, x="Tenure", y="survival",
                  color_discrete_sequence=[t["accent_blue"]], markers=True)
    fig.update_layout(
        **_layout(t),
        height=300,
        xaxis_title="Tenure bucket (months)", yaxis_title="Avg Survival Rate",
        showlegend=False, xaxis_tickangle=-30, yaxis_tickformat=".0%",
    )
    return fig