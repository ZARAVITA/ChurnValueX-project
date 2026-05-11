"""
ui/tabs/tab_data_guide.py — Full-page premium Data Guide.
Replaces the old sidebar expander "À propos des données".
"""

import streamlit as st
import pandas as pd
from utils.synthetic import generate_synthetic_data
from utils.export import to_excel_bytes


COLUMN_DOC_SHORT = [
    ("CustomerID",          "int",   "Identifiant client unique"),
    ("Churn",               "int",   "1 = churn / 0 = retenu"),
    ("Tenure",              "float", "Ancienneté client (mois)"),
    ("SatisfactionScore",   "int",   "Niveau de satisfaction (1–5)"),
    ("OrderCount",          "int",   "Nombre de commandes"),
    ("DaySinceLastOrder",   "int",   "Jours depuis la dernière commande"),
    ("CashbackAmount",      "float", "Valeur cashback reçue"),
    ("PreferredPaymentMode","str",   "Méthode de paiement préférée"),
]

COLUMN_DOC_FULL = [
    ("CustomerID",                  "int",   "Identifiant unique client"),
    ("Churn",                       "int",   "Cible : 1 = churné, 0 = retenu"),
    ("Tenure",                      "float", "Ancienneté en mois"),
    ("CityTier",                    "int",   "Niveau de ville : 1 (métropole) → 3 (rural)"),
    ("WarehouseToHome",             "int",   "Distance entrepôt → domicile (km)"),
    ("HourSpendOnApp",              "float", "Heures/semaine sur l'app ou site"),
    ("NumberOfDeviceRegistered",    "int",   "Appareils enregistrés"),
    ("SatisfactionScore",           "int",   "Score satisfaction (1–5)"),
    ("NumberOfAddress",             "int",   "Adresses de livraison sauvegardées"),
    ("Complain",                    "int",   "1 = réclamation déposée ce mois"),
    ("OrderAmountHikeFromlastYear", "float", "Hausse % du montant commandé vs an dernier"),
    ("CouponUsed",                  "int",   "Coupons utilisés ce mois"),
    ("OrderCount",                  "int",   "Commandes passées ce mois"),
    ("DaySinceLastOrder",           "int",   "Jours depuis la dernière commande"),
    ("CashbackAmount",              "float", "Cashback reçu ce mois (€/$)"),
    ("Gender",                      "str",   "Male / Female"),
    ("PreferredLoginDevice",        "str",   "Mobile Phone / Computer / Phone"),
    ("PreferredPaymentMode",        "str",   "Debit Card / Credit Card / UPI / …"),
    ("PreferedOrderCat",            "str",   "Laptop & Accessory / Mobile / Fashion / …"),
    ("MaritalStatus",               "str",   "Married / Single / Divorced"),
]

USE_CASES = [
    ("🔍", "Détection précoce du churn",             "Identifiez les signaux faibles avant qu'un client ne parte."),
    ("💎", "Priorisation à forte valeur",            "Concentrez les efforts sur les clients générant le plus de revenu."),
    ("💰", "Analyse du revenu exposé",               "Quantifiez le pipeline financier à risque sur 12 mois."),
    ("🎯", "Segmentation marketing intelligente",    "4 quadrants actionnables pour des stratégies différenciées."),
    ("📊", "Pilotage CRM",                           "Intégrez directement les scores dans votre CRM/CDP."),
    ("📩", "Optimisation campagnes rétention",       "Améliorez le ROI en ciblant les bons clients au bon moment."),
]

COMPATIBLE = ["CRM", "ERP", "E-commerce", "Retail", "Abonnements", "Services digitaux", "Plateformes B2B"]


def _col_table_html(rows, t):
    cells = ""
    for name, typ, desc in rows:
        cells += f"""
        <tr>
            <td style="font-family:'DM Mono',monospace;color:{t['accent_blue']};
                       font-size:0.78rem;white-space:nowrap;padding:0.6rem 0.9rem;
                       border-bottom:1px solid {t['border']}">{name}</td>
            <td style="color:{t['text_muted']};font-size:0.72rem;padding:0.6rem 0.9rem;
                       border-bottom:1px solid {t['border']};font-family:'DM Mono',monospace">{typ}</td>
            <td style="color:{t['text_secondary']};font-size:0.82rem;padding:0.6rem 0.9rem;
                       border-bottom:1px solid {t['border']}">{desc}</td>
        </tr>"""
    return f"""
<table style="width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden">
<thead>
  <tr style="background:{t['bg_card2']}">
    <th style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
               text-transform:uppercase;color:{t['text_muted']};padding:0.6rem 0.9rem;
               text-align:left;border-bottom:1px solid {t['border_strong']}">Variable</th>
    <th style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
               text-transform:uppercase;color:{t['text_muted']};padding:0.6rem 0.9rem;
               text-align:left;border-bottom:1px solid {t['border_strong']}">Type</th>
    <th style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
               text-transform:uppercase;color:{t['text_muted']};padding:0.6rem 0.9rem;
               text-align:left;border-bottom:1px solid {t['border_strong']}">Description</th>
  </tr>
</thead>
<tbody>{cells}</tbody>
</table>"""


def render(t: dict):
    # ── Back button ───────────────────────────────────────────────────────────
    if st.button("← Retour au dashboard", key="guide_back"):
        st.session_state["show_data_guide"] = False
        st.rerun()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{t['bg_card']} 0%,{t['bg_card2']} 100%);
            border:1px solid {t['border']};border-radius:20px;padding:2.4rem 3rem;
            margin:1rem 0 2rem;position:relative;overflow:hidden">
    <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
                background:radial-gradient(circle,{t['accent_blue_bg']} 0%,transparent 70%);
                pointer-events:none"></div>
    <div style="display:inline-block;background:{t['accent_blue_bg']};border:1px solid {t['accent_blue_bd']};
                color:{t['accent_blue']};padding:0.25rem 0.9rem;border-radius:100px;font-size:0.62rem;
                font-family:'DM Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;
                margin-bottom:1rem">📖 Documentation</div>
    <h1 style="font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;
               color:{t['text_primary']};margin:0 0 0.5rem;line-height:1.1">
        Guide des données
    </h1>
    <p style="font-size:1rem;color:{t['text_secondary']};max-width:620px;line-height:1.6;margin:0">
        Structure recommandée pour l'analyse intelligente du churn et de la valeur client.
    </p>
</div>
""", unsafe_allow_html=True)

    # ── Compatible with ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Compatibilité</div>', unsafe_allow_html=True)
    badges = "".join([
        f'<span style="display:inline-block;background:{t["accent_blue_bg"]};'
        f'border:1px solid {t["accent_blue_bd"]};color:{t["accent_blue"]};'
        f'padding:0.25rem 0.75rem;border-radius:100px;font-size:0.74rem;'
        f'font-weight:600;margin:0.2rem 0.3rem 0.2rem 0">{tag}</span>'
        for tag in COMPATIBLE
    ])
    st.markdown(f'<div style="margin-bottom:2rem">{badges}</div>', unsafe_allow_html=True)

    # ── Objectif système + Download ────────────────────────────────────────────
    g1, g2 = st.columns([3, 2])

    with g1:
        st.markdown(f"""
<div style="background:{t['bg_card']};border:1px solid {t['border']};border-radius:16px;
            padding:1.5rem 1.8rem;height:100%">
    <div class="sec-label">Objectif du système</div>
    <div style="font-size:1rem;font-weight:700;color:{t['text_primary']};margin-bottom:0.8rem">
        Transformer les données en décisions
    </div>
    <p style="font-size:0.85rem;color:{t['text_secondary']};line-height:1.7;margin:0">
        ChurnIQ transforme les données clients brutes en décisions business actionnables grâce à la
        <strong>prédiction du churn</strong>, l'<strong>analyse de la valeur client</strong>,
        l'identification du <strong>revenu à risque</strong>, et la
        <strong>segmentation décisionnelle</strong> en 4 quadrants.
    </p>
</div>
""", unsafe_allow_html=True)

    with g2:
        # Generate synthetic demo for download
        demo_df = generate_synthetic_data(900)
        excel_bytes = to_excel_bytes(demo_df)

        st.markdown(f"""
<div style="background:{t['bg_card']};border:1px solid {t['border']};border-radius:16px;
            padding:1.5rem 1.8rem;height:100%">
    <div class="sec-label">Dataset de démonstration</div>
    <div style="font-size:1rem;font-weight:700;color:{t['text_primary']};margin-bottom:0.6rem">
        📥 Télécharger le dataset Excel
    </div>
    <p style="font-size:0.8rem;color:{t['text_secondary']};line-height:1.6;margin:0 0 1rem">
        Dataset prêt à l'emploi pour CRM et e-commerce.<br>
        Modifiez-le et réimportez-le dans ChurnIQ.
    </p>
""", unsafe_allow_html=True)

        st.download_button(
            label="⬇  Télécharger le dataset Excel",
            data=excel_bytes,
            file_name="churniq_demo_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # ── Variables table ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Variables utilisées</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">📊 Variables principales</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="background:{t["bg_card"]};border:1px solid {t["border"]};'
        f'border-radius:14px;overflow:hidden;margin-bottom:1rem">'
        f'{_col_table_html(COLUMN_DOC_SHORT, t)}</div>',
        unsafe_allow_html=True
    )

    with st.expander("▶  Voir toutes les variables utilisées (20 colonnes)", expanded=False):
        st.markdown(
            f'<div style="background:{t["bg_card"]};border:1px solid {t["border"]};'
            f'border-radius:14px;overflow:hidden;margin-top:0.5rem">'
            f'{_col_table_html(COLUMN_DOC_FULL, t)}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # ── Adaptability ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Flexibilité</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">⚙️ Adaptabilité entreprise</div>', unsafe_allow_html=True)

    adapt_items = [
        ("Colonnes disponibles", "Le pipeline s'adapte aux colonnes présentes dans votre dataset."),
        ("Objectifs métier",     "Les seuils de segmentation sont reconfigurables selon vos KPIs."),
        ("KPIs internes",        "Ajoutez vos propres métriques de valeur et de risque."),
        ("Règles de scoring",    "Personnalisez les règles de la matrice de décision."),
        ("Contraintes spécifiques", "Intégration possible avec CRM, CDP, data warehouse."),
    ]
    adapt_cols = st.columns(len(adapt_items))
    for col, (title, desc) in zip(adapt_cols, adapt_items):
        col.markdown(f"""
<div style="background:{t['bg_card']};border:1px solid {t['border']};border-radius:12px;
            padding:1rem 1.1rem;height:100%">
    <div style="font-size:0.78rem;font-weight:700;color:{t['text_primary']};margin-bottom:0.35rem">{title}</div>
    <div style="font-size:0.75rem;color:{t['text_secondary']};line-height:1.55">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

    # ── Use cases ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Applications</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title">✓ Cas d\'utilisation typiques</div>', unsafe_allow_html=True)

    uc_cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(USE_CASES):
        with uc_cols[i % 3]:
            st.markdown(f"""
<div style="background:{t['bg_card']};border:1px solid {t['border']};border-radius:14px;
            padding:1.1rem 1.3rem;margin-bottom:0.8rem">
    <div style="font-size:1.3rem;margin-bottom:0.5rem">{icon}</div>
    <div style="font-size:0.85rem;font-weight:700;color:{t['text_primary']};margin-bottom:0.3rem">{title}</div>
    <div style="font-size:0.78rem;color:{t['text_secondary']};line-height:1.55">{desc}</div>
</div>""", unsafe_allow_html=True)