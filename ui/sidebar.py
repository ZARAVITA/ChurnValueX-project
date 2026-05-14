"""
ui/sidebar.py
"""
import streamlit as st
import pandas as pd
from config.theme import LIGHT, DARK


def _sidebar_extra_css(t):
    return f"""
<style>
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {t.get('bg_sidebar_top', t['bg_sidebar'])} 0%, {t['bg_card2']} 100%) !important;
    border-right: 1px solid {t['border']} !important;
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 0 !important;
    margin-top: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child {{
    padding-top: 0.4rem !important;
    margin-top: 0 !important;
}}
section[data-testid="stSidebar"] hr {{
    margin: 0.3rem 0 !important;
}}

/* ── Brand ── */
.sb-brand-name {{ font-size:1.2rem;font-weight:900;letter-spacing:-.03em;line-height:1; }}
.sb-brand-name .churn {{ color:#E53E3E !important; }}
.sb-brand-name .iq {{ color:#38B2AC !important; }}
.sb-brand-sub {{ font-size:.58rem;opacity:.4;font-family:'DM Mono',monospace;letter-spacing:.12em;text-transform:uppercase;color:{t['text_muted']} !important; }}

/* ── Titre INITIALISER ── */
.sb-init-title {{
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -.03em;
    line-height: 1.05;
    color: {t['text_primary']} !important;
    text-transform: uppercase;
    margin: .6rem 0 .2rem;
}}
.sb-init-sub {{
    font-size: .75rem;
    color: {t['text_secondary']} !important;
    line-height: 1.5;
    margin-bottom: .85rem;
}}

/* ── Options radio visuelles ── */
.sb-radio-wrap {{
    border: 1.5px dashed rgba(239,68,68,0.35);
    border-radius: 12px;
    padding: .55rem .6rem;
    margin-bottom: .55rem;
}}
.sb-radio-option {{
    display: flex;
    align-items: center;
    gap: .65rem;
    padding: .65rem .75rem;
    border-radius: 10px;
    border: 1px solid {t['border']};
    background: {t['bg_card']};
    transition: all .18s ease;
    margin-bottom: .35rem;
}}
.sb-radio-option.selected {{
    border-color: {t['accent_blue']} !important;
    background: {t['accent_blue_bg']} !important;
}}
.sb-radio-option:last-child {{ margin-bottom: 0; }}
.sb-radio-dot {{
    width: 18px; height: 18px;
    border-radius: 50%;
    border: 2px solid {t['border']};
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
}}
.sb-radio-dot.checked {{
    border-color: {t['accent_blue']};
    background: {t['accent_blue_bg']};
}}
.sb-radio-dot.checked::after {{
    content: '';
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {t['accent_blue']};
    display: block;
}}
.sb-radio-icon {{ font-size: 1.1rem; flex-shrink: 0; }}
.sb-radio-text {{
    flex: 1;
    font-size: .82rem;
    font-weight: 700;
    color: {t['text_primary']} !important;
    line-height: 1.3;
}}
.sb-radio-text small {{
    display: block;
    font-size: .68rem;
    font-weight: 400;
    color: {t['text_muted']} !important;
    margin-top: .1rem;
}}
.sb-radio-badge {{
    font-size: .58rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: {t['accent_blue']} !important;
    background: {t['accent_blue_bg']};
    border: 1px solid {t['accent_blue_bd']};
    padding: .18rem .5rem;
    border-radius: 100px;
    white-space: nowrap;
}}

/* ── Separateur OU ── */
.sb-ou {{
    display: flex; align-items: center; gap: 8px; margin: .4rem 0;
}}
.sb-ou-line {{ flex:1; height:1px; background:{t['border']}; opacity:.5; }}
.sb-ou-txt {{
    font-size: .7rem; font-weight: 800;
    font-family: 'DM Mono', monospace;
    letter-spacing: .1em;
    color: {t['text_muted']} !important;
    text-transform: uppercase;
}}

/* ── Boutons de selection (invisibles visuellement) ── */
section[data-testid="stSidebar"] .sb-sel-btns .stButton > button {{
    background: transparent !important;
    color: {t['text_muted']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
    font-size: .68rem !important;
    font-weight: 600 !important;
    padding: .3rem .5rem !important;
    transition: all .15s ease !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .sb-sel-btns .stButton > button:hover {{
    border-color: {t['accent_blue']} !important;
    color: {t['accent_blue']} !important;
    transform: none !important;
    opacity: 1 !important;
}}

/* ── Zone upload conditionnelle ── */
.sb-upload-zone {{
    border: 1.5px dashed {t['accent_blue_bd']};
    border-radius: 12px;
    padding: .55rem .7rem .45rem;
    background: {t['accent_blue_bg']};
    margin-bottom: .55rem;
}}
.sb-upload-hint {{
    font-size: .68rem;
    color: {t['accent_blue']} !important;
    font-family: 'DM Mono', monospace;
    text-align: center;
    margin-bottom: .3rem;
    line-height: 1.5;
}}
.sb-upload-hint strong {{ color: {t['accent_blue']} !important; }}

/* ── Bouton lancer principal ── */
section[data-testid="stSidebar"] .launch-btn-wrap .stButton > button,
section[data-testid="stSidebar"] .launch-upload-btn-wrap .stButton > button {{
    background: #1E40AF !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: .86rem !important;
    font-weight: 700 !important;
    padding: .72rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: -.01em !important;
    box-shadow: 0 4px 14px rgba(30,64,175,.35) !important;
    transition: all .22s ease !important;
}}
section[data-testid="stSidebar"] .launch-btn-wrap .stButton > button:hover,
section[data-testid="stSidebar"] .launch-upload-btn-wrap .stButton > button:hover {{
    box-shadow: 0 6px 20px rgba(30,64,175,.55) !important;
    transform: translateY(-1px) !important;
    opacity: 1 !important;
}}

/* ── File uploader ── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{ border:1.5px dashed {t['border']} !important;border-radius:10px !important;background:{t['bg_card2']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {{ border-color:{t['accent_blue']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{ color:{t['text_secondary']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background:transparent !important;border:none !important;min-height:0 !important;padding:.4rem .6rem !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {{ display:none !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{ font-size:.66rem !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{ padding:.28rem .8rem !important;font-size:.74rem !important;border-radius:7px !important; }}

/* ── Parametres entreprises ── */
.sb-ent-toggle {{ display:none; }}
.sb-ent-btn {{ display:flex;align-items:center;justify-content:space-between;width:100%;padding:.6rem .9rem;background:{t['bg_card']} !important;border:1px solid {t['border']};border-radius:10px;cursor:pointer;transition:all .18s ease;font-family:'Plus Jakarta Sans',sans-serif;margin-top:.35rem; }}
.sb-ent-btn:hover {{ background:{t['accent_blue_bg']} !important;border-color:{t['accent_blue_bd']}; }}
.sb-ent-label {{ font-size:.8rem;font-weight:700;color:{t['text_primary']} !important; }}
.sb-ent-chevron {{ font-size:.6rem;color:{t['text_muted']} !important;transition:transform .2s ease; }}
.sb-ent-body {{ display:none;border:1px solid {t['border']};border-top:none;border-radius:0 0 10px 10px;background:{t['bg_card2']};padding:.85rem 1rem .95rem;margin-top:-2px; }}
.sb-ent-toggle:checked ~ .sb-ent-btn .sb-ent-chevron {{ transform:rotate(180deg); }}
.sb-ent-toggle:checked ~ .sb-ent-btn {{ border-radius:10px 10px 0 0;border-bottom-color:transparent;background:{t['accent_blue_bg']} !important; }}
.sb-ent-toggle:checked ~ .sb-ent-body {{ display:block; }}
.sb-ent-desc {{ font-size:.72rem;color:{t['text_secondary']} !important;line-height:1.6;margin-bottom:.5rem; }}
.sb-ent-list {{ list-style:none;padding:0;margin:0 0 .6rem;display:flex;flex-direction:column;gap:.28rem; }}
.sb-ent-list li {{ font-size:.71rem;color:{t['text_muted']} !important;display:flex;align-items:center;gap:5px; }}
.sb-ent-list li::before {{ content:'\u2192';color:{t['accent_blue']} !important;font-weight:700;font-size:.68rem; }}
.sb-ent-cta {{ font-size:.69rem;color:{t['text_secondary']} !important;line-height:1.55;margin-bottom:.6rem;padding:.5rem .6rem;background:{t['accent_blue_bg']};border:1px solid {t['accent_blue_bd']};border-radius:7px; }}
.sb-ent-email {{ display:inline-block;font-size:.68rem;font-family:'DM Mono',monospace;color:{t['accent_blue']} !important;text-decoration:none;padding:.28rem .65rem;background:{t['accent_blue_bg']};border:1px solid {t['accent_blue_bd']};border-radius:7px;letter-spacing:.02em; }}

/* ── Status ── */
.status-active {{ margin-top:.7rem;padding:.6rem .9rem;background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.28);border-radius:10px;font-size:.74rem;color:#16A34A;display:flex;align-items:center;gap:8px; }}
.status-dot {{ width:6px;height:6px;border-radius:50%;background:#22C55E;animation:pulse-green 2s ease-in-out infinite;flex-shrink:0; }}
@keyframes pulse-green {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.55;transform:scale(.8)}} }}

/* ── Toggle theme ── */
.theme-icon-btn .stButton > button {{
    background: {t['bg_card2']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 50% !important;
    width: 34px !important; height: 34px !important;
    min-height: 0 !important;
    padding: 0 !important;
    font-size: 1rem !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: all .18s ease !important;
    box-shadow: none !important;
}}
.theme-icon-btn .stButton > button:hover {{
    background: {t['accent_blue_bg']} !important;
    border-color: {t['accent_blue_bd']} !important;
    transform: scale(1.12) !important;
    opacity: 1 !important;
}}
section[data-testid="stSidebar"] .theme-icon-btn {{
    display: flex !important;
    justify-content: center !important;
    margin: .3rem 0 .5rem !important;
}}
section[data-testid="stSidebar"] .theme-icon-btn .stButton {{
    display: flex !important;
    justify-content: center !important;
}}

/* Boutons sidebar primaires */
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] * {{
    background: #1E40AF !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(30,64,175,.35) !important;
    opacity: 1 !important;
}}
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {{
    background: #708090 !important;
    color: #ffffff !important;
    border: 1px solid #5a6a78 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}

/* ══════════════════════════════════════════════════
   DOCUMENTATION SECTION — séparation + bouton discret
══════════════════════════════════════════════════ */
.sb-doc-section {{
    margin-top: 22px;
    padding-top: 16px;
    border-top: 1px solid {t['border']};
}}
.sb-doc-label {{
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #475569 !important;
    margin-bottom: 7px;
    display: block;
    font-family: 'DM Mono', monospace;
    opacity: 0.68;
}}
section[data-testid="stSidebar"] .sb-doc-btn-wrap .stButton > button {{
    background: {t['bg_card']} !important;
    color: {t['text_secondary']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    font-size: .79rem !important;
    font-weight: 500 !important;
    padding: .46rem .9rem !important;
    width: 100% !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transition: background .18s ease, border-color .18s ease !important;
    opacity: 1 !important;
    transform: none !important;
}}
section[data-testid="stSidebar"] .sb-doc-btn-wrap .stButton > button:hover {{
    background: {t['bg_card2']} !important;
    border-color: {t['border_strong']} !important;
    color: {t['text_primary']} !important;
    box-shadow: none !important;
    transform: none !important;
    opacity: 1 !important;
}}
</style>
"""


def render_sidebar():
    with st.sidebar:
        dark_mode = st.session_state.get("dark_mode", True)
        theme = DARK if dark_mode else LIGHT

        st.markdown(_sidebar_extra_css(theme), unsafe_allow_html=True)

        # ── Brand header ──────────────────────────────────────────────────────
        st.markdown(
            '<div style="padding:.1rem 0 .15rem">'
            '<div class="sb-brand-name"><span class="churn">Churn</span><span class="iq">IQ</span></div>'
            '<div class="sb-brand-sub">Customer Intelligence Platform</div>'
            '</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<hr style="margin:.2rem 0 .4rem;border-color:{theme["border"]}">',
            unsafe_allow_html=True)

        # ── Titre principal grand + majuscules ────────────────────────────────
        st.markdown(
            '<div class="sb-init-title">Initialiser l\'analyse client</div>'
            '<div class="sb-init-sub">Lancez la d\u00e9mo int\u00e9gr\u00e9e ou importez vos propres donn\u00e9es.</div>',
            unsafe_allow_html=True)

        raw_df = st.session_state.get("raw_df", None)

        # ── Etat selection ────────────────────────────────────────────────────
        if "data_source" not in st.session_state:
            st.session_state["data_source"] = "demo"

        data_source  = st.session_state["data_source"]
        opt1_selected = data_source == "demo"
        opt2_selected = data_source == "upload"

        # ── Rendu visuel des deux options radio ───────────────────────────────
        opt1_cls = "sb-radio-option selected" if opt1_selected else "sb-radio-option"
        opt2_cls = "sb-radio-option selected" if opt2_selected else "sb-radio-option"
        dot1_cls = "sb-radio-dot checked"     if opt1_selected else "sb-radio-dot"
        dot2_cls = "sb-radio-dot checked"     if opt2_selected else "sb-radio-dot"
        badge1   = "<span class='sb-radio-badge'>S\u00c9LECTIONN\u00c9</span>" if opt1_selected else ""

        st.markdown(f"""
<div class="sb-radio-wrap">
  <div class="{opt1_cls}">
    <div class="{dot1_cls}"></div>
    <div class="sb-radio-icon">&#128196;</div>
    <div class="sb-radio-text">
      Base client CRM/e-commerce int\u00e9gr\u00e9e (D\u00e9mo)
      <small>( 5630 clients)</small>
    </div>
    {badge1}
  </div>
  <div class="sb-ou">
    <div class="sb-ou-line"></div>
    <span class="sb-ou-txt">OU</span>
    <div class="sb-ou-line"></div>
  </div>
  <div class="{opt2_cls}">
    <div class="{dot2_cls}"></div>
    <div class="sb-radio-icon">&#11014;</div>
    <div class="sb-radio-text">Importez vos donn\u00e9es clients/CRM (Upload)</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Boutons de selection compacts sous le bloc radio
        st.markdown('<div class="sb-sel-btns">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            lbl1 = "\u2713 D\u00e9mo" if opt1_selected else "D\u00e9mo"
            if st.button(lbl1, key="sel_demo", use_container_width=True):
                st.session_state["data_source"] = "demo"
                st.rerun()
        with c2:
            lbl2 = "\u2713 Upload" if opt2_selected else "Upload"
            if st.button(lbl2, key="sel_upload", use_container_width=True):
                st.session_state["data_source"] = "upload"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Zone upload — visible uniquement si option 2 ──────────────────────
        if opt2_selected:
            st.markdown(
                f'<div style="background:{theme["bg_card2"]};border:1.5px dashed {theme["border"]}; '
                f'border-radius:10px 10px 0 0;padding:.55rem .9rem .35rem;'
                f'border-bottom:none;margin-bottom:0;">'
                f'<div style="font-size:.78rem;font-weight:600;color:{theme["text_secondary"]};'
                f'display:flex;align-items:center;gap:6px;">'
                f'\U0001f4c2 D\u00e9posez vos donn\u00e9es ici</div>'
                f'<div style="font-size:.62rem;color:{theme["text_muted"]};font-family:\'DM Mono\',monospace;'
                f'letter-spacing:.04em;margin-top:.15rem;">CSV, Excel \u2022 Jusqu\u2019\u00e0 200&nbsp;MB</div>'
                f'</div>',
                unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "upload", type=["xlsx", "csv"],
                help="CSV, Excel \u2022 Jusqu\u2019\u00e0 200 MB",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                try:
                    _df = (pd.read_csv(uploaded_file)
                           if uploaded_file.name.endswith(".csv")
                           else pd.read_excel(uploaded_file))
                    st.session_state["uploaded_df"]   = _df
                    st.session_state["uploaded_name"] = uploaded_file.name
                except Exception as e:
                    st.error(f"Erreur lecture : {e}")
                    st.session_state.pop("uploaded_df", None)

            if st.session_state.get("uploaded_df") is not None:
                _n    = len(st.session_state["uploaded_df"])
                _name = st.session_state.get("uploaded_name", "fichier")
                st.markdown(
                    f'<div style="font-size:.69rem;color:{theme["protect_text"]};'
                    f'font-family:\'DM Mono\',monospace;padding:.2rem .1rem;margin-top:.1rem">'
                    f'\u2713 {_name} \u2014 {_n:,} lignes</div>',
                    unsafe_allow_html=True)

        # ── Bouton principal unique — adaptatif selon le contexte ─────────────
        has_file = st.session_state.get("uploaded_df") is not None

        # Determine label + action selon le mode
        if opt1_selected:
            # CAS 1 : démo intégrée
            btn_label    = "\u25b6  Lancer une simulation business (D\u00e9mo)"
            btn_disabled = False
        elif opt2_selected and has_file:
            # CAS 2 : upload avec fichier valide
            btn_label    = "\u25b6  Lancer l\u2019analyse pr\u00e9dictive"
            btn_disabled = False
        else:
            # CAS 3 : upload sans fichier — bouton désactivé avec message
            btn_label    = "\u25b6  Importez un fichier pour lancer l\u2019analyse"
            btn_disabled = True

        # Style du bouton désactivé
        if btn_disabled:
            st.markdown(f"""
<style>
section[data-testid="stSidebar"] .launch-btn-wrap .stButton > button:disabled,
section[data-testid="stSidebar"] .launch-btn-wrap .stButton > button[disabled] {{
    background: {theme['bg_card2']} !important;
    color: {theme['text_muted']} !important;
    border: 1px solid {theme['border']} !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 0.6 !important;
    transform: none !important;
}}
</style>
""", unsafe_allow_html=True)

        st.markdown('<div class="launch-btn-wrap" style="margin-top:.65rem">', unsafe_allow_html=True)
        if st.button(
            btn_label,
            use_container_width=True,
            key="btn_main_launch",
            type="primary",
            disabled=btn_disabled
        ):
            if opt1_selected:
                # Action démo intégrée
                try:
                    raw_df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name=1)
                    st.session_state["raw_df"]       = raw_df
                    st.session_state["pipeline_key"] = None
                    st.success(f"\u2713 {len(raw_df):,} clients charg\u00e9s")
                except Exception:
                    from utils.synthetic import generate_synthetic_data
                    raw_df = generate_synthetic_data(900)
                    st.session_state["raw_df"]       = raw_df
                    st.session_state["pipeline_key"] = None
                    st.info("Fichier absent \u2014 dataset synth\u00e9tique g\u00e9n\u00e9r\u00e9.")
            elif opt2_selected and has_file:
                # Action analyse données importées
                raw_df = st.session_state["uploaded_df"]
                st.session_state["raw_df"]       = raw_df
                st.session_state["pipeline_key"] = None
                st.session_state.pop("uploaded_df", None)
                st.success(f"\u2713 {len(raw_df):,} clients import\u00e9s")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Statut dataset actif ──────────────────────────────────────────────
        if raw_df is not None:
            st.markdown(
                '<div class="status-active"><div class="status-dot"></div>'
                f'<div>Dataset actif \u2014 <strong>{len(raw_df):,} clients</strong></div></div>',
                unsafe_allow_html=True)

        # ── Paramètres entreprises ────────────────────────────────────────────
        st.markdown(
            '<div class="sb-ent-wrap">'
            '<input type="checkbox" class="sb-ent-toggle" id="sb_ent_acc">'
            '<label class="sb-ent-btn" for="sb_ent_acc">'
            '<span class="sb-ent-label">Param\u00e8tres entreprises</span>'
            '<span class="sb-ent-chevron">&#9660;</span>'
            '</label>'
            '<div class="sb-ent-body">'
            '<p class="sb-ent-desc">Donn&eacute;es diff&eacute;rentes de la structure standard\u00a0? '
            'ChurnIQ peut &ecirc;tre adapt&eacute; &agrave; votre organisation.</p>'
            '<ul class="sb-ent-list">'
            '<li>Vos donn&eacute;es CRM</li>'
            '<li>Vos indicateurs business</li>'
            '<li>Votre segmentation client</li>'
            '<li>Vos r&egrave;gles de scoring</li>'
            '<li>Vos objectifs m&eacute;tier</li>'
            '</ul>'
            '<p class="sb-ent-cta">ChurnIQ n\'est qu\'un exemple de ce qu\'il est possible de construire. '
            'Contactez-nous pour cr&eacute;er des solutions intelligentes adapt&eacute;es '
            '&agrave; vos propres donn&eacute;es et enjeux m&eacute;tier.</p>'
            '<a class="sb-ent-email" href="mailto:zaravitamds18@gmail.com">&#128233; zaravitamds18@gmail.com</a>'
            '</div></div>',
            unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # DOCUMENTATION — séparation visuelle claire + bouton ressource secondaire
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            '<div class="sb-doc-section">'
            '<span class="sb-doc-label">Documentation</span>'
            '</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sb-doc-btn-wrap">', unsafe_allow_html=True)
        if st.button(
            "\U0001f4d8  Guide d\u2019int\u00e9gration des donn\u00e9es",
            use_container_width=True,
            key="btn_guide"
        ):
            st.session_state["show_data_guide"] = True
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── Toggle dark/light — bouton rond centré ────────────────────────────
        icon_btn = "\U0001f319" if not dark_mode else "\u2600\ufe0f"
        st.markdown('<div class="theme-icon-btn">', unsafe_allow_html=True)
        if st.button(icon_btn, key="dark_toggle"):
            st.session_state["dark_mode"] = not dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:.62rem;opacity:.28;font-family:\'DM Mono\',monospace;'
            'line-height:1.9;margin-top:.3rem;text-align:center;">'
            'ZARA VITA<br>\u00a9 2025 ChurnIQ \u2013 Tous droits r\u00e9serv\u00e9s'
            '</div>',
            unsafe_allow_html=True)

    return raw_df, theme, st.session_state.get("show_data_guide", False)