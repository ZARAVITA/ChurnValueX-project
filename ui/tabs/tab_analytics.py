"""
ui/tabs/tab_analytics.py — Tab 2: Deep analytics — churn drivers, survival, CLV by segment.
"""

import streamlit as st
from ui.charts import (fig_tenure_churn, fig_satisfaction_churn,
                       fig_clv_by_segment, fig_survival_curves, fig_seg_pie)


def render(df, t):
    st.markdown('<div class="sec-label">Analyse approfondie</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Indicateurs & Facteurs de Churn</div>', unsafe_allow_html=True)

    # Row 1 — Tenure vs churn + satisfaction vs churn
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Risque de churn par ancienneté client**")
        st.plotly_chart(fig_tenure_churn(df, t), use_container_width=True, config={"displayModeBar": False})
    with r1c2:
        st.markdown("**Risque de churn par score de satisfaction**")
        st.plotly_chart(fig_satisfaction_churn(df, t), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # Row 2 — CLV by segment + pie
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**CLV médiane par segment**")
        st.plotly_chart(fig_clv_by_segment(df, t), use_container_width=True, config={"displayModeBar": False})
    with r2c2:
        st.markdown("**Répartition des clients par segment**")
        st.plotly_chart(fig_seg_pie(df, t), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # Row 3 — Survival curve
    st.markdown("**Courbe de survie moyenne par ancienneté**")
    st.plotly_chart(fig_survival_curves(df, t), use_container_width=True, config={"displayModeBar": False})

    # Row 4 — Feature stats table
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown("**Statistiques descriptives par segment**")

    num_cols = ["CLV_12mois", "Churn_proba", "proba_churn_1mois", "proba_churn_3mois"]
    num_cols = [c for c in num_cols if c in df.columns]

    grp = df.groupby("Matrix_Segment")[num_cols].agg(["mean", "median"]).round(3)
    grp.columns = [" — ".join(c) for c in grp.columns]
    st.dataframe(grp, use_container_width=True)
