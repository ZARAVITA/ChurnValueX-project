"""
ui/sidebar.py — Sidebar: theme toggle, data loader, data-about section.
"""

import streamlit as st
import pandas as pd
from config.theme import LIGHT, DARK


COLUMN_DOC = [
    ("CustomerID",                    "int",    "Unique customer identifier"),
    ("Churn",                         "int",    "Target: 1 = churned, 0 = retained"),
    ("Tenure",                        "float",  "Months since customer's first order"),
    ("CityTier",                      "int",    "City tier: 1 (metro) → 3 (rural)"),
    ("WarehouseToHome",               "int",    "Distance warehouse → home (km)"),
    ("HourSpendOnApp",                "float",  "Avg hours/week on app or website"),
    ("NumberOfDeviceRegistered",      "int",    "Devices registered on the account"),
    ("SatisfactionScore",             "int",    "Satisfaction survey score (1–5)"),
    ("NumberOfAddress",               "int",    "Number of saved delivery addresses"),
    ("Complain",                      "int",    "1 = complaint filed in last month"),
    ("OrderAmountHikeFromlastYear",   "float",  "% increase in order amount vs last year"),
    ("CouponUsed",                    "int",    "Coupons used in last month"),
    ("OrderCount",                    "int",    "Orders placed in last month"),
    ("DaySinceLastOrder",             "int",    "Days since last order"),
    ("CashbackAmount",                "float",  "Cashback received in last month (€/$)"),
    ("Gender",                        "str",    "Male / Female"),
    ("PreferredLoginDevice",          "str",    "Mobile Phone / Computer / Phone"),
    ("PreferredPaymentMode",          "str",    "Debit Card / Credit Card / UPI / …"),
    ("PreferedOrderCat",              "str",    "Laptop & Accessory / Mobile / Fashion / …"),
    ("MaritalStatus",                 "str",    "Married / Single / Divorced"),
]


def render_sidebar() -> tuple:
    """
    Returns (raw_df, theme_dict).
    raw_df is None if no data has been loaded yet.
    """
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:0.6rem 0 1.2rem">
            <div style="font-size:1.15rem;font-weight:800;letter-spacing:-0.02em">⬡ ChurnIQ</div>
            <div style="font-size:0.7rem;opacity:0.5;font-family:'DM Mono',monospace;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px">
                Customer Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Theme toggle ───────────────────────────────────────────────────────
        dark_mode = st.toggle("🌙  Dark mode", value=st.session_state.get("dark_mode", False))
        # Only update dark_mode in session state — never clear pipeline data
        st.session_state["dark_mode"] = dark_mode
        theme = DARK if dark_mode else LIGHT

        st.divider()

        # ── Data loader ────────────────────────────────────────────────────────
        st.markdown("**📂 Données**")

        raw_df = st.session_state.get("raw_df", None)

        if st.button("▶  Charger la démo intégrée", use_container_width=True):
            try:
                raw_df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name=1)
                st.session_state["raw_df"] = raw_df
                # Clear pipeline cache so it reruns with the new dataset
                st.session_state.pop("pipeline_df",  None)
                st.session_state.pop("pipeline_key", None)
                st.success(f"✓ {len(raw_df):,} clients chargés")
            except Exception:
                from utils.synthetic import generate_synthetic_data
                raw_df = generate_synthetic_data(900)
                st.session_state["raw_df"] = raw_df
                # Clear pipeline cache
                st.session_state.pop("pipeline_df",  None)
                st.session_state.pop("pipeline_key", None)
                st.info("Fichier non trouvé — dataset synthétique généré (900 clients).")

        st.divider()

        # ── Data about ─────────────────────────────────────────────────────────
        with st.expander("ℹ️  À propos des données", expanded=False):
            st.markdown("""
**Structure attendue**

Ce système est conçu pour des **données clients e-commerce / CRM** structurées.
Une entreprise dispose généralement de ces informations issues de son ERP, CRM ou plateforme de commandes.

Le dataset d'exemple contient **20 variables** décrivant le profil, le comportement
et l'historique de chaque client.

> ✅ **Adaptable** : les colonnes et seuils peuvent être reconfigurés pour s'aligner
> sur les données spécifiques de chaque entreprise.
            """)

            st.markdown("---")
            st.markdown("**Colonnes & signification**")

            rows = ""
            for name, typ, desc in COLUMN_DOC:
                rows += f"<tr><td class='col-name'>{name}</td><td class='col-type'>{typ}</td><td>{desc}</td></tr>"

            st.markdown(f"""
<table class="col-table">
<thead><tr><th>Colonne</th><th>Type</th><th>Description</th></tr></thead>
<tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

        # ── Status ─────────────────────────────────────────────────────────────
        if raw_df is not None:
            pipeline_ready = "pipeline_df" in st.session_state
            status_color = "#16A34A" if pipeline_ready else "#D97706"
            status_bg    = "rgba(34,197,94,0.12)"  if pipeline_ready else "rgba(217,119,6,0.12)"
            status_bd    = "rgba(34,197,94,0.3)"   if pipeline_ready else "rgba(217,119,6,0.3)"
            status_text  = f"✓ Dataset actif — {len(raw_df):,} clients" if pipeline_ready else f"⟳ {len(raw_df):,} clients — analyse en cours..."
            st.markdown(f"""
<div style="margin-top:0.8rem;padding:0.7rem 1rem;background:{status_bg};
border:1px solid {status_bd};border-radius:10px;font-size:0.78rem;color:{status_color}">
{status_text}
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
<div style="font-size:0.68rem;opacity:0.4;font-family:'DM Mono',monospace;line-height:1.8">
ZARA VITA<br>Smart Automation Technologies<br>zaravitamds18@gmail.com
</div>
""", unsafe_allow_html=True)

    return raw_df, theme