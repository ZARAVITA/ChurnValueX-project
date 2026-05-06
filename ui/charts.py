"""
ui/charts.py — All Plotly figures, theme-aware.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.theme import SEG_COLORS_LIGHT, SEG_COLORS_DARK


def _layout(t: dict) -> dict:
    return dict(
        paper_bgcolor=t["plotly_paper"],
        plot_bgcolor =t["plotly_plot"],
        font=dict(family="Plus Jakarta Sans, sans-serif", color=t["plotly_font"], size=12),
        xaxis=dict(gridcolor=t["plotly_grid"], zerolinecolor=t["plotly_grid"],
                   linecolor=t["plotly_grid"]),
        yaxis=dict(gridcolor=t["plotly_grid"], zerolinecolor=t["plotly_grid"],
                   linecolor=t["plotly_grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=35, b=10),
    )

def _seg_colors(t: dict) -> dict:
    return SEG_COLORS_DARK if t.get("bg_main","").startswith("#0") else SEG_COLORS_LIGHT


# ── Decision Matrix ───────────────────────────────────────────────────────────

def fig_decision_matrix(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    fig = px.scatter(
        df, x="CLV_12mois", y="Churn_proba",
        color="Matrix_Segment",
        color_discrete_map=sc,
        hover_data={"CustomerID": True, "CLV_12mois": ":.1f",
                    "Churn_proba": ":.1%", "Matrix_Segment": True},
        labels={"CLV_12mois": "Customer Lifetime Value (12M)", "Churn_proba": "Churn Risk"},
        opacity=0.72,
    )
    clv_med = df["CLV_12mois"].median()
    fig.add_hline(y=0.5,     line=dict(color="rgba(128,128,128,0.3)", dash="dot", width=1.5))
    fig.add_vline(x=clv_med, line=dict(color="rgba(128,128,128,0.3)", dash="dot", width=1.5))
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(**_layout(t), height=430,
                      yaxis_tickformat=".0%",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ── Before / After demo ───────────────────────────────────────────────────────

def fig_demo_before(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.scatter(df, x="CLV_12mois", y="Churn_proba",
                     color_discrete_sequence=["#94A3B8"],
                     labels={"CLV_12mois": "CLV", "Churn_proba": "Churn Risk"})
    fig.update_traces(marker=dict(size=6, opacity=0.4))
    fig.update_layout(**_layout(t), height=320,
                      title=dict(text="Raw data — no structure", font=dict(size=12)),
                      yaxis_tickformat=".0%")
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
    fig.update_layout(**_layout(t), height=320,
                      title=dict(text="4 action zones identified", font=dict(size=12)),
                      yaxis_tickformat=".0%",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ── Distributions ─────────────────────────────────────────────────────────────

def fig_clv_hist(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.histogram(df, x="CLV_12mois", nbins=50, color_discrete_sequence=[t["accent_blue"]])
    fig.update_layout(**_layout(t), height=270,
                      xaxis_title="CLV (12 months)", yaxis_title="Customers", showlegend=False)
    return fig


def fig_risk_hist(df: pd.DataFrame, t: dict) -> go.Figure:
    fig = px.histogram(df, x="Churn_proba", nbins=50,
                       color_discrete_sequence=["#F59E0B"])
    fig.update_layout(**_layout(t), height=270,
                      xaxis_title="Churn Probability", yaxis_title="Customers",
                      showlegend=False, xaxis_tickformat=".0%")
    return fig


def fig_seg_pie(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    counts = df["Matrix_Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.pie(counts, names="Segment", values="Count",
                 color="Segment", color_discrete_map=sc, hole=0.55)
    fig.update_layout(**_layout(t), height=280,
                      legend=dict(orientation="h", yanchor="top", y=-0.05))
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def fig_tenure_churn(df: pd.DataFrame, t: dict) -> go.Figure:
    if "Tenure" not in df.columns:
        return go.Figure()
    bins = pd.cut(df["Tenure"], bins=10)
    grp  = df.groupby(bins)["Churn_proba"].mean().reset_index()
    grp["Tenure"] = grp["Tenure"].astype(str)
    fig = px.bar(grp, x="Tenure", y="Churn_proba",
                 color_discrete_sequence=[t["accent_blue"]])
    fig.update_layout(**_layout(t), height=270,
                      xaxis_title="Tenure (months)", yaxis_title="Avg Churn Risk",
                      xaxis_tickangle=-30, showlegend=False,
                      yaxis_tickformat=".0%")
    return fig


def fig_satisfaction_churn(df: pd.DataFrame, t: dict) -> go.Figure:
    if "SatisfactionScore" not in df.columns:
        return go.Figure()
    grp = df.groupby("SatisfactionScore")["Churn_proba"].mean().reset_index()
    fig = px.bar(grp, x="SatisfactionScore", y="Churn_proba",
                 color_discrete_sequence=["#F59E0B"])
    fig.update_layout(**_layout(t), height=270,
                      xaxis_title="Satisfaction Score (1–5)", yaxis_title="Avg Churn Risk",
                      showlegend=False, yaxis_tickformat=".0%")
    return fig


def fig_clv_by_segment(df: pd.DataFrame, t: dict) -> go.Figure:
    sc = _seg_colors(t)
    grp = df.groupby("Matrix_Segment")["CLV_12mois"].median().reset_index()
    fig = px.bar(grp, x="Matrix_Segment", y="CLV_12mois",
                 color="Matrix_Segment", color_discrete_map=sc)
    fig.update_layout(**_layout(t), height=270,
                      xaxis_title="Segment", yaxis_title="Median CLV ($)",
                      showlegend=False)
    return fig


def fig_survival_curves(df: pd.DataFrame, t: dict) -> go.Figure:
    """Avg survival (1 - churn_proba) by tenure bucket."""
    if "Tenure" not in df.columns or "proba_churn_now" not in df.columns:
        return go.Figure()
    df2 = df.copy()
    df2["survival"] = 1 - df2["proba_churn_now"]
    bins = pd.cut(df2["Tenure"], bins=12)
    grp  = df2.groupby(bins)["survival"].mean().reset_index()
    grp["Tenure"] = grp["Tenure"].astype(str)
    fig = px.line(grp, x="Tenure", y="survival",
                  color_discrete_sequence=[t["accent_blue"]],
                  markers=True)
    fig.update_layout(**_layout(t), height=300,
                      xaxis_title="Tenure bucket (months)", yaxis_title="Avg Survival Rate",
                      showlegend=False, xaxis_tickangle=-30, yaxis_tickformat=".0%")
    return fig
