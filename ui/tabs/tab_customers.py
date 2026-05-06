"""
ui/tabs/tab_customers.py — Tab 4: Filterable, sortable customer list + recommendations.
"""

import streamlit as st
import numpy as np
from utils.export import to_excel_bytes


def _recommend(row):
    seg = row["Matrix_Segment"]
    if seg == "PRIORITY": return "🔴 Appel urgent + offre exclusive"
    if seg == "PROTECT":  return "🟢 Programme VIP"
    if seg == "OPTIMIZE": return "🟡 Email + coupon ciblé"
    return "⚫ Automation réengagement"


def render(df, t):
    st.markdown('<div class="sec-label">Intelligence client</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Liste Interactive des Clients</div>', unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        seg_filter = st.multiselect(
            "Filtrer par segment",
            ["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"],
            default=["PRIORITY", "PROTECT"],
        )
    with f2:
        clv_max = float(df["CLV_12mois"].quantile(0.99)) or 1.0
        clv_range = st.slider("CLV range ($)", 0.0, clv_max, (0.0, clv_max), step=0.5)
    with f3:
        sort_by = st.selectbox(
            "Trier par",
            {"Churn_proba": "Risque churn",
             "CLV_12mois": "CLV (12M)",
             "SPA_score": "Score priorité",
             "proba_churn_1mois": "Risque 1 mois"}.keys(),
            format_func=lambda x: {"Churn_proba":"Risque churn","CLV_12mois":"CLV (12M)",
                                    "SPA_score":"Score priorité","proba_churn_1mois":"Risque 1 mois"}[x]
        )

    # Risk level filter
    risk_thresh = st.slider("Seuil de risque minimum", 0.0, 1.0, 0.0, 0.05,
                            format="%.0f%%", help="Affiche uniquement les clients avec risque ≥ ce seuil")

    # ── Apply filters ─────────────────────────────────────────────────────────
    df_view = df.copy()
    df_view["Action recommandée"] = df_view.apply(_recommend, axis=1)

    mask = (
        df_view["Matrix_Segment"].isin(seg_filter) &
        df_view["CLV_12mois"].between(clv_range[0], clv_range[1]) &
        (df_view["Churn_proba"] >= risk_thresh)
    )
    df_view = df_view[mask].sort_values(sort_by, ascending=False)

    col_map = {
        "CustomerID":       "ID Client",
        "CLV_12mois":       "CLV 12M ($)",
        "Churn_proba":      "Risque",
        "proba_churn_1mois":"Risque 1M",
        "proba_churn_3mois":"Risque 3M",
        "Matrix_Segment":   "Segment",
        "Action recommandée":"Action",
    }
    available = [c for c in col_map if c in df_view.columns]
    df_display = df_view[available].rename(columns=col_map).copy()

    for pct_col in ["Risque", "Risque 1M", "Risque 3M"]:
        if pct_col in df_display.columns:
            df_display[pct_col] = df_display[pct_col].map(lambda x: f"{x:.1%}")
    if "CLV 12M ($)" in df_display.columns:
        df_display["CLV 12M ($)"] = df_display["CLV 12M ($)"].round(1)

    st.markdown(
        f'<div style="font-size:0.8rem;color:{t["text_muted"]};margin-bottom:0.5rem">'
        f'{len(df_display):,} clients correspondent aux filtres</div>',
        unsafe_allow_html=True
    )
    st.dataframe(df_display.head(300), use_container_width=True, height=420)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Téléchargements</div>', unsafe_allow_html=True)

    export_cols = [c for c in ["CustomerID","CLV_12mois","Churn_proba",
                                "proba_churn_1mois","proba_churn_3mois",
                                "Matrix_Segment","SPA_score"] if c in df.columns]
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        data = to_excel_bytes(df[df["Matrix_Segment"] == "PRIORITY"][export_cols].sort_values("SPA_score", ascending=False))
        st.download_button("⬇  Clients PRIORITY", data=data,
                           file_name="priority_customers.xlsx", use_container_width=True)
    with dl2:
        data2 = to_excel_bytes(df.sort_values("proba_churn_1mois", ascending=False)[export_cols].head(500))
        st.download_button("⬇  À risque (30 jours)", data=data2,
                           file_name="risk_1month.xlsx", use_container_width=True)
    with dl3:
        data3 = to_excel_bytes(df.sort_values("CLV_12mois", ascending=False)[export_cols].head(500))
        st.download_button("⬇  Top CLV clients", data=data3,
                           file_name="top_clv.xlsx", use_container_width=True)
