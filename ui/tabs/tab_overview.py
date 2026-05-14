"""
ui/tabs/tab_overview.py — Tab 1: Hero banner + KPI metrics + before/after demo.
"""

import streamlit as st
import numpy as np
from ui.charts import fig_demo_before, fig_demo_after, fig_clv_hist, fig_risk_hist


def render(df, t):
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-badge">⬡ Enterprise Analytics Platform</div>
    <div class="hero-title">Customer Churn<br><span>Intelligence System</span></div>
    <div class="hero-subtitle">
        Identifiez quels clients sont à risque, quel montant de revenus est exposé, 
        et quelles actions entreprendre  -  le tout dans une couche de décision unifiée.
    </div>
</div>
""", unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total        = len(df)
    high_risk_n  = (df["Churn_proba"] >= 0.5).sum()
    high_risk_pct= high_risk_n / total
    revenue_risk = df.loc[df["Churn_proba"] >= 0.5, "CLV_12mois"].sum()
    avg_clv      = df["CLV_12mois"].mean()
    top20_n      = max(1, int(total * 0.2))
    top20_rev_pct= (df.nlargest(top20_n, "CLV_12mois")["CLV_12mois"].sum()
                    / df["CLV_12mois"].sum()) if df["CLV_12mois"].sum() > 0 else 0

    k1, k2, k3, k4 = st.columns(4)

    # (gradient top bar, border color, value color)
    KPI_STYLES = [
        ("#EF4444, #DC2626",  "rgba(239,68,68,0.55)",   "#F87171"),   # rouge  — risque churn
        ("#F97316, #EA580C",  "rgba(249,115,22,0.55)",  "#FB923C"),   # orange — revenu exposé
        ("#3B82F6, #2563EB",  "rgba(59,130,246,0.45)",  "#60A5FA"),   # bleu   — concentration
        ("#10B981, #059669",  "rgba(16,185,129,0.45)",  "#34D399"),   # vert   — CLV moyen
    ]

    def kpi(col, label, value, delta, grad, border, val_color):
        col.markdown(f"""
<div style="
    background: {t['bg_card']};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    height: 100%;
">
    <div style="
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, {grad});
        border-radius: 16px 16px 0 0;
    "></div>
    <div style="
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {t['text_muted']};
        margin-bottom: 0.6rem;
    ">{label}</div>
    <div style="
        font-size: 2rem;
        font-weight: 800;
        color: {val_color};
        line-height: 1;
        letter-spacing: -0.02em;
    ">{value}</div>
    <div style="
        font-size: 0.76rem;
        color: {t['text_secondary']};
        margin-top: 0.55rem;
        font-family: 'DM Mono', monospace;
        opacity: 1;
        font-weight: 500;
    ">{delta}</div>
</div>""", unsafe_allow_html=True)

    kpi(k1, "Clients à haut risque",   f"{high_risk_pct:.1%}",  f"{high_risk_n:,} clients sur {total:,}",  *KPI_STYLES[0])
    kpi(k2, "Revenu exposé (CLV 12M)", f"${revenue_risk:,.0f}", "pipeline à risque de départ",              *KPI_STYLES[1])
    kpi(k3, "Concentration top-20%",   f"{top20_rev_pct:.1%}",  "de la CLV totale dans le top quintile",    *KPI_STYLES[2])
    kpi(k4, "CLV moyen client (12M)",  f"${avg_clv:,.1f}",      "valeur projetée par client",               *KPI_STYLES[3])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distributions ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Distributions</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.plotly_chart(fig_clv_hist(df, t), use_container_width=True, config={"displayModeBar": False})
    with dc2:
        st.plotly_chart(fig_risk_hist(df, t), use_container_width=True, config={"displayModeBar": False})

    # ── Before / After ────────────────────────────────────────────────────────
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Impact du système</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Avant vs Après l\'intelligence client</div>', unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    with bc1:
        st.plotly_chart(fig_demo_before(df, t), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="demo-label">SANS SYSTÈME — données brutes, aucune visibilité</div>', unsafe_allow_html=True)
    with bc2:
        st.plotly_chart(fig_demo_after(df, t), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="demo-label">AVEC SYSTÈME — 4 zones d\'action identifiées</div>', unsafe_allow_html=True)