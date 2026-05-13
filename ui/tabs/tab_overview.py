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

    def kpi(col, label, value, delta=""):
        col.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-delta">{delta}</div>
</div>""", unsafe_allow_html=True)

    kpi(k1, "Clients à haut risque",     f"{high_risk_pct:.1%}",  f"{high_risk_n:,} clients sur {total:,}")
    kpi(k2, "Revenu exposé (CLV 12M)",   f"${revenue_risk:,.0f}", "pipeline à risque de départ")
    kpi(k3, "Concentration top-20%",     f"{top20_rev_pct:.1%}",  "de la CLV totale dans le top quintile")
    kpi(k4, "CLV moyen client (12M)",    f"${avg_clv:,.1f}",      "valeur projetée par client")

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
