"""
ui/tabs/tab_matrix.py — Tab 3: Decision matrix — 4-quadrant plot + segment action cards.
"""

import streamlit as st
from ui.charts import fig_decision_matrix


SEG_META = {
    "PRIORITY": {
        "cls": "mc-priority", "color_key": "priority_text",
        "icon": "🔴", "label": "PRIORITY",
        "action": "Contact personnel immédiat + offre exclusive — client rentable en danger.",
    },
    "PROTECT": {
        "cls": "mc-protect", "color_key": "protect_text",
        "icon": "🟢", "label": "PROTECT",
        "action": "Programme de fidélité VIP — maintenir l'engagement, consolider la relation.",
    },
    "OPTIMIZE": {
        "cls": "mc-optimize", "color_key": "optimize_text",
        "icon": "🟡", "label": "OPTIMIZE",
        "action": "Campagne email + coupon ciblé — vérifier le ROI avant d'agir.",
    },
    "AUTOMATE": {
        "cls": "mc-automate", "color_key": "automate_text",
        "icon": "⚫", "label": "AUTOMATE",
        "action": "Réengagement automatisé à faible coût — ne pas sur-investir.",
    },
}


def render(df, t):
    st.markdown('<div class="sec-label">Segmentation stratégique</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Matrice de Décision Client</div>', unsafe_allow_html=True)

    st.markdown("""
<p style="font-size:0.88rem;color:#64748B;margin-bottom:1.2rem;max-width:680px">
Chaque client est positionné selon deux axes : sa <strong>valeur long terme (CLV)</strong>
et son <strong>risque de départ</strong>. Les quatre quadrants définissent quatre stratégies distinctes.
</p>
""", unsafe_allow_html=True)

    st.plotly_chart(fig_decision_matrix(df, t), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Actions par Segment</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for seg, col in zip(["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"], cols):
        meta   = SEG_META[seg]
        seg_df = df[df["Matrix_Segment"] == seg]
        rev    = seg_df["CLV_12mois"].sum()
        cnt    = len(seg_df)
        color  = t[meta["color_key"]]

        col.markdown(f"""
<div class="{meta['cls']}">
    <div class="mc-label" style="color:{color}">{meta['icon']} {meta['label']}</div>
    <div class="mc-count" style="color:{color}">{cnt:,}</div>
    <div class="mc-revenue">Revenu : ${rev:,.0f}</div>
    <div class="mc-action">{meta['action']}</div>
</div>""", unsafe_allow_html=True)
