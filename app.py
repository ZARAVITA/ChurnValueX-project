# -*- coding: utf-8 -*-
"""
ChurnIQ — Customer Churn Intelligence System
Entry point: streamlit run app.py
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
from ui.sidebar import render_sidebar
from ui.styles  import build_css
from models.pipeline import run_pipeline

import ui.tabs.tab_overview  as tab_overview
import ui.tabs.tab_analytics as tab_analytics
import ui.tabs.tab_matrix    as tab_matrix
import ui.tabs.tab_customers as tab_customers
import ui.tabs.tab_insights  as tab_insights
import ui.tabs.tab_data_guide as tab_data_guide

# ── Sidebar ───────────────────────────────────────────────────────────────────
raw_df, theme, show_data_guide = render_sidebar()

# ── Apply global CSS ──────────────────────────────────────────────────────────
st.markdown(build_css(theme), unsafe_allow_html=True)


def build_welcome(t):
    block = (
        "<style>"
        "@keyframes wcUp {"
        "  from{opacity:0;transform:translateY(18px)}"
        "  to  {opacity:1;transform:translateY(0)}"
        "}"
        "@keyframes wcPulse {"
        "  0%,100%{opacity:1} 50%{opacity:.35}"
        "}"
        "@keyframes wcDrop {"
        "  from{opacity:0;transform:translateY(-8px)}"
        "  to  {opacity:1;transform:translateY(0)}"
        "}"
        ".wcRoot{"
        "  max-width:960px;margin:1.8rem auto 0;"
        "  padding:0 0 3rem;"
        "  font-family:'Plus Jakarta Sans',sans-serif;"
        "}"
        ".wcHero{"
        "  background:linear-gradient(140deg,__BG_CARD__ 0%,__BG_CARD2__ 100%);"
        "  border:1px solid __BORDER__;"
        "  border-radius:18px;"
        "  padding:2.2rem 2.8rem 2rem;"
        "  position:relative;overflow:hidden;"
        "  animation:wcUp .45s ease both;"
        "  margin-bottom:.9rem;"
        "}"
        ".wcHero::before{"
        "  content:'';"
        "  position:absolute;top:-90px;right:-90px;"
        "  width:340px;height:340px;"
        "  background:radial-gradient(circle,__ACCENT_BG__ 0%,transparent 65%);"
        "  pointer-events:none;"
        "}"
        ".wcBadge{"
        "  display:inline-flex;align-items:center;gap:.4rem;"
        "  background:__ACCENT_BG__;border:1px solid __ACCENT_BD__;"
        "  color:__ACCENT__ !important;"
        "  padding:.22rem .8rem;border-radius:100px;"
        "  font-size:.61rem;font-family:'DM Mono',monospace;"
        "  letter-spacing:.13em;text-transform:uppercase;"
        "  margin-bottom:1rem;position:relative;z-index:1;"
        "}"
        ".wcDot{"
        "  width:5px;height:5px;background:__ACCENT__;"
        "  border-radius:50%;animation:wcPulse 2.2s ease-in-out infinite;"
        "}"
        ".wcHeroBody{"
        "  display:grid;grid-template-columns:1fr auto;"
        "  gap:1.8rem;align-items:start;"
        "  position:relative;z-index:1;"
        "}"
        ".wcTitle{"
        "  font-size:2.3rem;font-weight:800;"
        "  letter-spacing:-.04em;line-height:1.08;"
        "  color:__TEXT_PRIMARY__ !important;margin-bottom:.7rem;"
        "}"
        ".wcAccent{"
        "  background:linear-gradient(118deg,__ACCENT__ 0%,__ACCENT2__ 100%);"
        "  -webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "  background-clip:text;"
        "}"
        ".wcSub{"
        "  font-size:.95rem;font-weight:600;"
        "  color:__TEXT_PRIMARY__ !important;"
        "  line-height:1.55;max-width:540px;margin-bottom:.45rem;"
        "}"
        ".wcDesc{"
        "  font-size:.84rem;color:__TEXT_SECONDARY__ !important;"
        "  line-height:1.7;max-width:510px;"
        "}"
        ".wcStats{display:flex;flex-direction:column;gap:.6rem;min-width:148px;}"
        ".wcStatPill{"
        "  background:__BG_CARD__;border:1px solid __BORDER__;"
        "  border-radius:11px;padding:.55rem .9rem;text-align:center;"
        "}"
        ".wcStatVal{"
        "  font-size:1.3rem;font-weight:800;"
        "  color:__ACCENT__ !important;letter-spacing:-.03em;line-height:1;"
        "}"
        ".wcStatLbl{"
        "  font-size:.6rem;font-family:'DM Mono',monospace;"
        "  letter-spacing:.08em;text-transform:uppercase;"
        "  color:__TEXT_MUTED__ !important;margin-top:.15rem;"
        "}"
        ".wcCols{"
        "  display:grid;grid-template-columns:1fr 1fr;"
        "  gap:.9rem;margin-bottom:.9rem;"
        "  animation:wcUp .45s .1s ease both;animation-fill-mode:both;"
        "}"
        ".wcCard{"
        "  background:__BG_CARD__;border:1px solid __BORDER__;"
        "  border-radius:15px;padding:1.5rem 1.7rem;"
        "}"
        ".wcSecLbl{"
        "  font-size:.59rem;font-family:'DM Mono',monospace;"
        "  letter-spacing:.14em;text-transform:uppercase;"
        "  color:__ACCENT__ !important;margin-bottom:.9rem;"
        "  display:flex;align-items:center;gap:.38rem;"
        "}"
        ".wcSecLbl::before{"
        "  content:'';display:inline-block;"
        "  width:5px;height:5px;background:__ACCENT__;border-radius:50%;"
        "}"
        ".wcCapList{display:flex;flex-direction:column;gap:.5rem;}"
        ".wcCapItem{"
        "  display:flex;align-items:center;gap:.55rem;"
        "  font-size:.855rem;color:__TEXT_SECONDARY__ !important;"
        "}"
        ".wcChk{"
        "  width:17px;height:17px;border-radius:5px;"
        "  background:__ACCENT_BG__;border:1px solid __ACCENT_BD__;"
        "  display:flex;align-items:center;justify-content:center;"
        "  flex-shrink:0;font-size:.62rem;"
        "  color:__ACCENT__ !important;font-weight:700;"
        "}"
        ".wcInfoCard{"
        "  background:__BG_CARD__;border:1px solid __BORDER__;"
        "  border-radius:15px;padding:1.5rem 1.7rem;"
        "  display:flex;flex-direction:column;gap:1rem;"
        "}"
        ".wcInfoRow{display:flex;align-items:flex-start;gap:.7rem;}"
        ".wcInfoIco{"
        "  width:30px;height:30px;border-radius:8px;"
        "  background:__ACCENT_BG__;border:1px solid __ACCENT_BD__;"
        "  display:flex;align-items:center;justify-content:center;"
        "  flex-shrink:0;font-size:.8rem;"
        "}"
        ".wcInfoTxt{font-size:.83rem;color:__TEXT_SECONDARY__ !important;line-height:1.6;}"
        ".wcInfoTxt strong{color:__TEXT_PRIMARY__ !important;}"
        ".wcHr{border:none;border-top:1px solid __BORDER__;margin:0;}"
        ".wcExpWrap{"
        "  animation:wcUp .45s .18s ease both;animation-fill-mode:both;"
        "}"
        ".wcExpBtn{"
        "  width:100%;background:__BG_CARD__;border:1px solid __BORDER__;"
        "  border-radius:13px;"
        "  padding:.95rem 1.4rem;"
        "  display:flex;align-items:center;justify-content:space-between;"
        "  cursor:pointer;"
        "  transition:border-color .18s,background .18s,transform .18s;"
        "  text-align:left;font-family:'Plus Jakarta Sans',sans-serif;"
        "  position:relative;z-index:2;"
        "}"
        ".wcExpBtn:hover{"
        "  border-color:__ACCENT_BD__;background:__ACCENT_BG__;"
        "  transform:translateY(-1px);"
        "}"
        ".wcExpLeft{display:flex;align-items:center;gap:.8rem;}"
        ".wcTag{"
        "  font-family:'DM Mono',monospace;font-size:.68rem;"
        "  letter-spacing:.06em;color:__ACCENT__ !important;"
        "  background:__ACCENT_BG__;border:1px solid __ACCENT_BD__;"
        "  padding:.2rem .6rem;border-radius:6px;"
        "}"
        ".wcExpLabel{"
        "  font-size:.88rem;font-weight:700;"
        "  color:__TEXT_PRIMARY__ !important;line-height:1.2;"
        "}"
        ".wcExpSub{"
        "  font-size:.72rem;color:__TEXT_MUTED__ !important;"
        "  font-family:'DM Mono',monospace;letter-spacing:.04em;margin-top:.1rem;"
        "}"
        ".wcChevBox{"
        "  width:26px;height:26px;border-radius:7px;"
        "  background:__ACCENT_BG__;border:1px solid __ACCENT_BD__;"
        "  display:flex;align-items:center;justify-content:center;"
        "  color:__ACCENT__ !important;font-size:.7rem;"
        "  transition:transform .22s ease;"
        "  flex-shrink:0;"
        "}"
        "#wcToggle{display:none;}"
        "#wcToggle:checked ~ .wcExpBody{"
        "  display:block;animation:wcDrop .22s ease both;"
        "}"
        "#wcToggle:checked ~ label.wcExpBtn .wcChevBox{"
        "  transform:rotate(180deg);"
        "}"
        "#wcToggle:checked ~ label.wcExpBtn{"
        "  border-radius:13px 13px 0 0;"
        "  border-bottom-color:transparent;"
        "}"
        ".wcExpBody{"
        "  display:none;"
        "  border:1px solid __BORDER__;border-top:none;"
        "  border-radius:0 0 13px 13px;"
        "  background:__BG_CARD2__;"
        "  padding:1.3rem 1.6rem 1.5rem;"
        "  margin-top:-2px;"
        "}"
        ".wcList{"
        "  list-style:none;margin:0;padding:0;"
        "  display:flex;flex-direction:column;gap:.55rem;"
        "  margin-bottom:.9rem;"
        "}"
        ".wcList li{"
        "  display:flex;align-items:flex-start;gap:.65rem;"
        "  font-size:.875rem;color:__TEXT_SECONDARY__ !important;line-height:1.5;"
        "}"
        ".wcListDot{"
        "  width:6px;height:6px;border-radius:50%;"
        "  background:__ACCENT__;flex-shrink:0;margin-top:.45rem;"
        "}"
        ".wcNote{"
        "  font-size:.82rem;color:__TEXT_MUTED__ !important;"
        "  line-height:1.7;padding-top:.75rem;"
        "  border-top:1px solid __BORDER__;"
        "  font-style:italic;"
        "}"
        "</style>"

        "<div class='wcRoot'>"
        "<div class='wcHero'>"
        "<div class='wcBadge'><span class='wcDot'></span>AI Decision System &middot; Enterprise Grade</div>"
        "<div class='wcHeroBody'>"
        "<div>"
        "<div class='wcTitle'>Bienvenue sur&nbsp;<span class='wcAccent'>ChurnIQ</span></div>"
        "<div class='wcSub'>Systeme decisionnel intelligent concu pour anticiper le churn, proteger votre revenu et maximiser la valeur client.</div>"
        "<div class='wcDesc'>Transformez vos donnees CRM en decisions business actionnables &mdash; identifiez les clients a risque et estimez le revenu expose en temps reel.</div>"
        "</div>"
        "<div class='wcStats'>"
        "<div class='wcStatPill'><div class='wcStatVal'>98%</div><div class='wcStatLbl'>Precision</div></div>"
        "<div class='wcStatPill'><div class='wcStatVal'>5</div><div class='wcStatLbl'>Dashboards</div></div>"
        "<div class='wcStatPill'><div class='wcStatVal'>12M</div><div class='wcStatLbl'>CLV horizon</div></div>"
        "</div>"
        "</div>"
        "</div>"
        "<div class='wcCols'>"
        "<div class='wcCard'>"
        "<div class='wcSecLbl'>Capacites intelligentes</div>"
        "<div class='wcCapList'>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Vision client 360&deg;</div>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Detection predictive du churn</div>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Analyse CLV &amp; revenu a risque</div>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Segmentation intelligente orientee decision</div>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Recommandations business exploitables</div>"
        "<div class='wcCapItem'><div class='wcChk'>&#10003;</div>Simulation d&rsquo;impact business</div>"
        "</div>"
        "</div>"
        "<div class='wcInfoCard'>"
        "<div class='wcInfoRow'>"
        "<div class='wcInfoIco'>&#9881;</div>"
        "<div class='wcInfoTxt'><strong>Adaptable</strong> aux donnees, KPIs et objectifs specifiques de chaque entreprise.</div>"
        "</div>"
        "<hr class='wcHr'>"
        "<div class='wcInfoRow'>"
        "<div class='wcInfoIco'>&#128196;</div>"
        "<div class='wcInfoTxt'><strong>Dataset de demo</strong> integre &mdash; ou synthetique (900 clients) si le fichier est absent.</div>"
        "</div>"
        "<hr class='wcHr'>"
        "<div class='wcInfoRow'>"
        "<div class='wcInfoIco'>&#9654;</div>"
        "<div class='wcInfoTxt'>Cliquez sur <strong>&#34;Lancer ChurnIQ avec les donnees de demonstration&#34;</strong> dans le panneau gauche pour activer le systeme.</div>"
        "</div>"
        "</div>"
        "</div>"
        "<div class='wcExpWrap'>"
        "<input type='checkbox' id='wcToggle'>"
        "<label class='wcExpBtn' for='wcToggle'>"
        "<div class='wcExpLeft'>"
        "<div class='wcTag'>Multi-Model AI</div>"
        "<div>"
        "<div class='wcExpLabel'>Intelligence analytique multi-modeles</div>"
        "<div class='wcExpSub'>Cliquez pour decouvrir l&rsquo;architecture IA</div>"
        "</div>"
        "</div>"
        "<div class='wcChevBox'>&#9660;</div>"
        "</label>"
        "<div class='wcExpBody'>"
        "<p style='font-size:.85rem;font-weight:700;color:__TEXT_PRIMARY__ !important;margin:0 0 .7rem;'>ChurnIQ combine :</p>"
        "<ul class='wcList'>"
        "<li><div class='wcListDot'></div>Intelligence predictive</li>"
        "<li><div class='wcListDot'></div>Analyse comportementale</li>"
        "<li><div class='wcListDot'></div>Modelisation CLV</li>"
        "<li><div class='wcListDot'></div>Analyse de survie</li>"
        "</ul>"
        "<p class='wcNote'>dans une interface exploitable par les equipes metier et decisionnelles.</p>"
        "</div>"
        "</div>"
        "</div>"
    )

    a  = t["accent_blue"]
    a2 = t["hero_grad_c"]
    block = (block
        .replace("__ACCENT_BG__",      t["accent_blue_bg"])
        .replace("__ACCENT_BD__",      t["accent_blue_bd"])
        .replace("__ACCENT2__",        a2)
        .replace("__ACCENT__",         a)
        .replace("__BG_CARD2__",       t["bg_card2"])
        .replace("__BG_CARD__",        t["bg_card"])
        .replace("__BORDER__",         t["border"])
        .replace("__TEXT_PRIMARY__",   t["text_primary"])
        .replace("__TEXT_SECONDARY__", t["text_secondary"])
        .replace("__TEXT_MUTED__",     t["text_muted"])
    )
    return block


# ── Data Guide page (full-screen, bypasses everything else) ───────────────────
if show_data_guide:
    tab_data_guide.render(theme)
    st.stop()

# ── No data state — Premium Welcome Screen ────────────────────────────────────
if raw_df is None:
    st.markdown(build_welcome(theme), unsafe_allow_html=True)
    st.stop()


# ── Dataset validation ───────────────────────────────────────────────────────
from utils.validator import validate_dataset

def _show_incompatible_error(result: dict, t: dict):
    """Render a premium, business-friendly incompatibility card."""
    # Pre-extract all values as plain variables — no backslashes inside f-string expressions (Python < 3.12)
    issues_html    = "".join("<li>" + i + "</li>" for i in result["issues"])
    warnings_html  = "".join("<li>" + w + "</li>" for w in result["warnings"]) if result["warnings"] else ""
    score_color    = "#EF4444" if result["score"] < 40 else "#F59E0B"
    present_count  = len(result["present"])
    total_count    = 20
    score_val      = result["score"]
    bg_card        = t["bg_card"]
    text_primary   = t["text_primary"]
    text_secondary = t["text_secondary"]
    text_muted     = t["text_muted"]
    accent_bg      = t["accent_blue_bg"]
    accent_bd      = t["accent_blue_bd"]
    border_col     = t["border"]

    # Warnings block — built as a plain string before the f-string
    if warnings_html:
        warnings_block = (
            '<div class="compat-issues-title" style="color:#F59E0B !important">'
            "Avertissements</div>"
            '<ul class="compat-warnings">' + warnings_html + "</ul>"
        )
    else:
        warnings_block = ""

    html = (
        "<style>"
        "@keyframes slideIn {"
        "  from { opacity:0; transform:translateY(12px); }"
        "  to   { opacity:1; transform:translateY(0); }"
        "}"
        f".compat-card {{ background:{bg_card};border:1.5px solid #FCA5A5;border-left:4px solid #EF4444;"
        "border-radius:16px;padding:1.8rem 2.2rem;max-width:720px;margin:2rem auto;"
        "animation:slideIn .35s ease both;font-family:'Plus Jakarta Sans',sans-serif;}"
        ".compat-header {display:flex;align-items:flex-start;gap:1rem;margin-bottom:1.2rem;}"
        ".compat-icon {width:42px;height:42px;border-radius:12px;background:rgba(239,68,68,0.12);"
        "border:1px solid rgba(239,68,68,0.25);display:flex;align-items:center;justify-content:center;"
        "font-size:1.2rem;flex-shrink:0;}"
        f".compat-title {{font-size:1.05rem;font-weight:800;color:{text_primary} !important;"
        "margin-bottom:0.3rem;letter-spacing:-0.02em;}"
        f".compat-sub {{font-size:0.82rem;color:{text_secondary} !important;line-height:1.5;}}"
        ".compat-score-row {display:flex;align-items:center;gap:0.8rem;margin-bottom:1.1rem;"
        "padding:0.75rem 1rem;background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.12);border-radius:10px;}"
        f".compat-score-val {{font-size:1.4rem;font-weight:900;color:{score_color} !important;"
        "font-family:'DM Mono',monospace;letter-spacing:-0.02em;line-height:1;}"
        f".compat-score-label {{font-size:0.75rem;color:{text_muted} !important;line-height:1.4;}}"
        ".compat-issues-title {font-size:0.72rem;font-family:'DM Mono',monospace;letter-spacing:0.1em;"
        "text-transform:uppercase;color:#EF4444 !important;margin-bottom:0.5rem;}"
        ".compat-issues {list-style:none;padding:0;margin:0 0 1rem;display:flex;flex-direction:column;gap:0.4rem;}"
        f".compat-issues li {{font-size:0.83rem;color:{text_secondary} !important;padding:0.45rem 0.75rem;"
        "background:rgba(239,68,68,0.06);border-radius:8px;border-left:2px solid #EF4444;line-height:1.45;}"
        ".compat-warnings {list-style:none;padding:0;margin:0 0 1rem;display:flex;flex-direction:column;gap:0.35rem;}"
        f".compat-warnings li {{font-size:0.82rem;color:{text_secondary} !important;padding:0.4rem 0.7rem;"
        "background:rgba(245,158,11,0.07);border-radius:8px;border-left:2px solid #F59E0B;}"
        f".compat-guide-box {{background:{accent_bg};border:1px solid {accent_bd};border-radius:12px;"
        "padding:1rem 1.2rem;margin-top:0.5rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}"
        f".compat-guide-text {{font-size:0.84rem;color:{text_secondary} !important;line-height:1.55;}}"
        f".compat-guide-text strong {{color:{text_primary} !important;}}"
        f".compat-divider {{border:none;border-top:1px solid {border_col};margin:1rem 0;}}"
        "</style>"
        "<div class='compat-card'>"
        "<div class='compat-header'>"
        "<div class='compat-icon'>&#9888;&#65039;</div>"
        "<div>"
        "<div class='compat-title'>Donn&eacute;es incompatibles avec ChurnIQ</div>"
        "<div class='compat-sub'>Le dataset import&eacute; ne correspond pas &agrave; la structure attendue par le syst&egrave;me. "
        "L'analyse ne peut pas &ecirc;tre lanc&eacute;e.</div>"
        "</div></div>"
        "<div class='compat-score-row'>"
        f"<div class='compat-score-val'>{score_val}%</div>"
        "<div class='compat-score-label'>"
        f"Score de compatibilit&eacute;<br><span style='font-family:DM Mono,monospace;font-size:0.68rem'>{present_count} / {total_count} colonnes reconnues</span>"
        "</div></div>"
        "<div class='compat-issues-title'>Probl&egrave;mes d&eacute;tect&eacute;s</div>"
        f"<ul class='compat-issues'>{issues_html}</ul>"
        f"{warnings_block}"
        "<hr class='compat-divider'>"
        "<div class='compat-guide-box'>"
        "<div class='compat-guide-text'>"
        "<strong>&#128216; Consultez le Guide des donn&eacute;es</strong><br>"
        "D&eacute;couvrez les colonnes requises, les formats compatibles et la structure recommand&eacute;e pour l'analyse ChurnIQ."
        "</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    # Guide button — outside markdown block
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        st.markdown(
            "<style>.open-guide-btn .stButton > button {"
            "background:#2563EB !important;color:#fff !important;border:none !important;"
            "border-radius:10px !important;font-size:0.86rem !important;font-weight:700 !important;"
            "padding:0.65rem 1.4rem !important;width:100% !important;"
            "box-shadow:0 4px 14px rgba(37,99,235,0.3) !important;transition:all 0.2s ease !important;}"
            "</style>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="open-guide-btn">', unsafe_allow_html=True)
        if st.button("📖  Ouvrir le Guide des données", use_container_width=True, key="btn_open_guide_err"):
            st.session_state["show_data_guide"] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Validate dataset before pipeline ────────────────────────────────────────
_validation = validate_dataset(raw_df)

if not _validation["ok"]:
    _show_incompatible_error(_validation, theme)
    # Clear bad dataset from session so user can reload
    st.session_state.pop("raw_df", None)
    st.session_state.pop("pipeline_df", None)
    st.session_state.pop("pipeline_key", None)
    st.stop()

# ── Run pipeline — only once per data load ───────────────────────────────────
try:
    _raw_key = f"{len(raw_df)}_{int(raw_df.iloc[0, 0])}"
except Exception:
    _raw_key = f"{len(raw_df)}_0"

if st.session_state.get("pipeline_key") != _raw_key:
    with st.spinner("ChurnIQ is thinking..."):
        my_bar = st.progress(0, text="Initializing ChurnIQ AI Engine...")
        try:
            for pct in range(100):
                time.sleep(0.03)
                if pct < 30:   _txt = "Analyzing customer behavior..."
                elif pct < 60: _txt = "Computing churn probabilities..."
                elif pct < 90: _txt = "Generating AI insights..."
                else:          _txt = "Finalizing intelligence dashboard..."
                my_bar.progress(pct + 1, text=_txt)
            st.session_state["pipeline_df"]  = run_pipeline(raw_df)
            st.session_state["pipeline_key"] = _raw_key
            my_bar.empty()
        except Exception as _pipeline_err:
            my_bar.empty()
            _show_incompatible_error({
                "ok": False, "score": 0, "present": [], "warnings": [],
                "issues": [
                    "Le pipeline analytique n'a pas pu traiter ce dataset.",
                    "Vérifiez que les types de données et les valeurs sont conformes au format attendu.",
                ]
            }, theme)
            st.session_state.pop("raw_df", None)
            st.session_state.pop("pipeline_df", None)
            st.session_state.pop("pipeline_key", None)
            st.stop()

df = st.session_state["pipeline_df"]


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
st.markdown(
    '<div class="footer">'
    'ChurnIQ &middot; Customer Churn Intelligence System<br>'
    'Developpe par <strong>ZARA VITA</strong> &mdash; Smart Automation Technologies &middot; '
    '<a href="mailto:zaravitamds18@gmail.com">zaravitamds18@gmail.com</a> &middot; +212 770 636 297'
    '</div>',
    unsafe_allow_html=True,
)