"""
ui/tabs/tab_insights.py — Tab 5: Consulting insights + what-if simulation.
"""

import streamlit as st
import numpy as np


def render(df, t):
    total     = len(df)
    total_rev = df["CLV_12mois"].sum()

    # ── Key insights ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Analyse consulting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Insights Clés</div>', unsafe_allow_html=True)

    priority_rev     = df[df["Matrix_Segment"] == "PRIORITY"]["CLV_12mois"].sum()
    priority_rev_pct = priority_rev / total_rev if total_rev > 0 else 0
    top20_n          = max(1, int(total * 0.2))
    top20_rev_pct    = (df.nlargest(top20_n, "CLV_12mois")["CLV_12mois"].sum()
                        / total_rev) if total_rev > 0 else 0
    high_risk_n      = (df["Churn_proba"] >= 0.5).sum()
    priority_n       = (df["Matrix_Segment"] == "PRIORITY").sum()

    insights = [
        (
            f"{priority_rev_pct:.0%}",
            f"du pipeline de revenu sur 12 mois est détenu par les clients <strong>PRIORITY</strong> — "
            f"des clients à haute valeur actuellement à risque de partir. Une intervention immédiate est requise "
            f"pour protéger <strong>${priority_rev:,.0f}</strong> de revenus projetés."
        ),
        (
            f"{top20_rev_pct:.0%}",
            f"de la CLV totale est concentrée dans les <strong>top 20%</strong> clients ({top20_n:,} comptes). "
            f"Perdre l'un d'eux a un impact financier disproportionné. "
            f"Ces clients méritent une attention particulière et des actions proactives."
        ),
        (
            f"{high_risk_n:,}",
            f"clients sont actuellement au-dessus du seuil de 50% de risque de churn. "
            f"Parmi eux, <strong>{priority_n:,}</strong> ont une CLV supérieure à la médiane. "
            f"Cibler le segment PRIORITY en priorité maximise le ROI par euro marketing dépensé."
        ),
    ]

    for stat, text in insights:
        st.markdown(f"""
<div class="insight-card">
    <span class="insight-stat">{stat}</span>
    <span class="insight-text">{text}</span>
</div>""", unsafe_allow_html=True)

    # ── Simulation ────────────────────────────────────────────────────────────
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Simulation stratégique</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Analyse What-If</div>', unsafe_allow_html=True)

    st.markdown("""
<p style="font-size:0.88rem;color:#64748B;margin-bottom:1rem;max-width:680px">
Simulez l'impact financier d'une réduction du taux de churn sur les clients PRIORITY.
Ajustez le curseur pour voir les revenus récupérables selon l'efficacité de vos actions de rétention.
</p>
""", unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns([2, 3])
    with sim_c1:
        reduction = st.slider("Réduction du churn sur PRIORITY (%)", 0, 60, 15, step=5)
        cost_per_action = st.number_input("Coût moyen par action de rétention ($)", value=25.0, step=5.0)

    priority_df      = df[df["Matrix_Segment"] == "PRIORITY"].copy()
    saved_n          = int(len(priority_df) * reduction / 100)
    saved_rev        = (priority_df.nlargest(saved_n, "CLV_12mois")["CLV_12mois"].sum()
                        if saved_n > 0 else 0)
    total_cost       = saved_n * cost_per_action
    net_gain         = saved_rev - total_cost
    roi              = (net_gain / total_cost * 100) if total_cost > 0 else 0

    with sim_c2:
        st.markdown(f"""
<div class="sim-box">
    <h4>Résultat : −{reduction}% churn sur le segment PRIORITY</h4>
    <p>
        <strong style="font-size:1.5rem">${saved_rev:,.0f}</strong> de revenus récupérés
        sur <strong>{saved_n:,} clients</strong> retenus.<br><br>
        Coût estimé des actions : <strong>${total_cost:,.0f}</strong><br>
        Gain net : <strong>${net_gain:,.0f}</strong><br>
        <strong>ROI estimé : {roi:,.0f}%</strong><br><br>
        <em style="font-size:0.8rem;opacity:0.7">
        Hypothèse : les actions de rétention s'appliquent en priorité aux clients à plus forte CLV.
        </em>
    </p>
</div>""", unsafe_allow_html=True)

    # ── Segment revenue breakdown table ───────────────────────────────────────
    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown("**Synthèse financière par segment**")

    summary_rows = []
    for seg in ["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"]:
        seg_df = df[df["Matrix_Segment"] == seg]
        summary_rows.append({
            "Segment":        seg,
            "Clients":        len(seg_df),
            "CLV total ($)":  round(seg_df["CLV_12mois"].sum(), 0),
            "CLV moyen ($)":  round(seg_df["CLV_12mois"].mean(), 1),
            "Risque moyen":   f"{seg_df['Churn_proba'].mean():.1%}",
            "Risque 1M moyen":f"{seg_df['proba_churn_1mois'].mean():.1%}" if "proba_churn_1mois" in seg_df else "N/A",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(summary_rows).set_index("Segment"),
                 use_container_width=True)
