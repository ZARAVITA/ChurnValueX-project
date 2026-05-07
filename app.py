# -*- coding: utf-8 -*-
"""
ChurnIQ — Customer Churn Intelligence System
Entry point: streamlit run app.py

Project structure:
  app.py              ← this file
  config/theme.py     ← light/dark design tokens
  ui/styles.py        ← CSS generator
  ui/sidebar.py       ← sidebar + theme toggle + data loader + data-about
  ui/charts.py        ← all Plotly figures
  ui/tabs/
    tab_overview.py   ← Tab 1: Hero + KPIs + before/after
    tab_analytics.py  ← Tab 2: Deep analytics
    tab_matrix.py     ← Tab 3: Decision matrix
    tab_customers.py  ← Tab 4: Customer list + export
    tab_insights.py   ← Tab 5: Insights + simulation
  models/pipeline.py  ← full ML pipeline (cached)
  utils/synthetic.py  ← synthetic data generator
  utils/export.py     ← Excel export helper
"""

import os
os.environ["STREAMLIT_USE_ARROW"] = "0"

import streamlit as st

st.set_page_config(
    page_title="ChurnIQ — Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⬡",
)
import time
# ── Imports (after page config) ───────────────────────────────────────────────
from ui.sidebar import render_sidebar
from ui.styles  import build_css
from models.pipeline import run_pipeline

import ui.tabs.tab_overview  as tab_overview
import ui.tabs.tab_analytics as tab_analytics
import ui.tabs.tab_matrix    as tab_matrix
import ui.tabs.tab_customers as tab_customers
import ui.tabs.tab_insights  as tab_insights

# ── Sidebar (returns raw_df + theme) ─────────────────────────────────────────
raw_df, theme = render_sidebar()

# ── Apply CSS ─────────────────────────────────────────────────────────────────
st.markdown(build_css(theme), unsafe_allow_html=True)

# ── No data state ─────────────────────────────────────────────────────────────
if raw_df is None:
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:60vh;text-align:center;color:{theme['text_muted']}">
    <div style="font-size:3rem;margin-bottom:1rem">⬡</div>
    <div style="font-size:1.2rem;font-weight:700;color:{theme['text_secondary']};margin-bottom:0.5rem">
        ChurnIQ est prêt
    </div>
    <div style="font-size:0.9rem;max-width:420px;line-height:1.6">
        Chargez le dataset de démonstration depuis la barre latérale pour activer le système d'intelligence client.
    </div>
    <div style="margin-top:1.5rem;font-family:'DM Mono',monospace;font-size:0.7rem;
                letter-spacing:0.1em;text-transform:uppercase;opacity:0.5">
        ← Utiliser le panneau gauche
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Run pipeline (only when data changes, not on filter interactions) ─────────
pipeline_key = f"pipeline_{id(raw_df)}_{len(raw_df)}"

if st.session_state.get("pipeline_key") != pipeline_key:
    with st.spinner("⬡  ChurnIQ is thinking..."):
        progress_text = "Initializing ChurnIQ AI Engine..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.03)
            if percent_complete < 30:
                text = "Analyzing customer behavior..."
            elif percent_complete < 60:
                text = "Computing churn probabilities..."
            elif percent_complete < 90:
                text = "Generating AI insights..."
            else:
                text = "Finalizing intelligence dashboard..."
            my_bar.progress(percent_complete + 1, text=text)

        df = run_pipeline(raw_df)
        my_bar.empty()

        st.session_state["pipeline_key"]    = pipeline_key
        st.session_state["pipeline_result"] = df
else:
    df = st.session_state["pipeline_result"]

# ── Main tabs ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Vue d'ensemble",
    "Indicateurs",
    "Matrice de décision",
    "Clients",
    "Insights & Simulation",
])

with tabs[0]:
    tab_overview.render(df, theme)

with tabs[1]:
    tab_analytics.render(df, theme)

with tabs[2]:
    tab_matrix.render(df, theme)

with tabs[3]:
    tab_customers.render(df, theme)

with tabs[4]:
    tab_insights.render(df, theme)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    ChurnIQ · Customer Churn Intelligence System<br>
    Développé par <strong>ZARA VITA</strong> — Smart Automation Technologies ·
    <a href="mailto:zaravitamds18@gmail.com">zaravitamds18@gmail.com</a> · +212 770 636 297
</div>
""", unsafe_allow_html=True)