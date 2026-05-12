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
section[data-testid="stSidebar"] .sb-brand-cols > div[data-testid="stHorizontalBlock"] {{
    gap: 0 !important; align-items: flex-start !important;
}}
section[data-testid="stSidebar"] .sb-brand-cols [data-testid="column"] {{ padding: 0 !important; }}
section[data-testid="stSidebar"] .sb-brand-cols [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] .sb-brand-cols .stVerticalBlock {{ gap: 0 !important; padding: 0 !important; }}
section[data-testid="stSidebar"] .sb-brand-cols .element-container {{ margin: 0 !important; padding: 0 !important; }}
.sb-brand-name {{ font-size:1.2rem;font-weight:900;letter-spacing:-.03em;color:{t['text_primary']} !important;line-height:1; }}
.sb-brand-sub {{ font-size:.58rem;opacity:.4;font-family:'DM Mono',monospace;letter-spacing:.12em;text-transform:uppercase;color:{t['text_muted']} !important; }}
section[data-testid="stSidebar"] .dark-toggle-wrap .stButton > button {{
    background:{t['bg_card2']} !important;border:1px solid {t['border']} !important;
    border-radius:7px !important;width:22px !important;height:22px !important;
    min-height:0 !important;padding:0 !important;display:flex !important;
    align-items:center !important;justify-content:center !important;
    font-size:.65rem !important;line-height:1 !important;color:{t['text_secondary']} !important;
    transition:all .18s ease !important;
}}
section[data-testid="stSidebar"] .dark-toggle-wrap .stButton > button:hover {{
    background:{t['accent_blue_bg']} !important;border-color:{t['accent_blue_bd']} !important;opacity:1 !important;
}}
.sb-section-title {{
    font-size:.82rem;font-weight:800;letter-spacing:-.01em;
    color:#1E40AF !important;
    margin-bottom:.2rem;display:block;
}}
.sb-section-sub {{ font-size:.72rem;color:{t['text_muted']};line-height:1.5;margin-bottom:.75rem;display:block; }}
section[data-testid="stSidebar"] .launch-btn-wrap .stButton > button,
section[data-testid="stSidebar"] .launch-upload-btn-wrap .stButton > button {{
    background: #1E40AF !important;
    color: #000000 !important;
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
.demo-badge {{ display:flex;align-items:center;gap:5px;font-size:.68rem;color:{t['text_muted']};margin-top:.35rem;font-family:'DM Mono',monospace;letter-spacing:.02em; }}
.demo-dot {{ width:4px;height:4px;border-radius:50%;background:{t['accent_blue']};opacity:.55;flex-shrink:0;display:inline-block; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{ border:1.5px dashed {t['border']} !important;border-radius:10px !important;background:{t['bg_card2']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {{ border-color:{t['accent_blue']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{ color:{t['text_secondary']} !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background:transparent !important;border:none !important;min-height:0 !important;padding:.4rem .6rem !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {{ display:none !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{ font-size:.66rem !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{ padding:.28rem .8rem !important;font-size:.74rem !important;border-radius:7px !important; }}
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
.status-active {{ margin-top:.7rem;padding:.6rem .9rem;background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.28);border-radius:10px;font-size:.74rem;color:#16A34A;display:flex;align-items:center;gap:8px; }}
.status-dot {{ width:6px;height:6px;border-radius:50%;background:#22C55E;animation:pulse-green 2s ease-in-out infinite;flex-shrink:0; }}
@keyframes pulse-green {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.55;transform:scale(.8)}} }}
</style>
"""


def render_sidebar():
    with st.sidebar:
        dark_mode = st.session_state.get("dark_mode", False)
        theme = DARK if dark_mode else LIGHT
        icon = "\U0001f319" if not dark_mode else "\u2600\ufe0f"

        st.markdown(_sidebar_extra_css(theme), unsafe_allow_html=True)

        # ── Brand header ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-brand-cols">', unsafe_allow_html=True)
        _bc, _tc = st.columns([6, 1])
        with _bc:
            st.markdown(
                '<div style="padding:.7rem 0 .4rem">'
                '<div class="sb-brand-name">\u29c1 ChurnIQ</div>'
                '<div class="sb-brand-sub">Customer Intelligence Platform</div>'
                '</div>', unsafe_allow_html=True)
        with _tc:
            st.markdown('<div class="dark-toggle-wrap" style="padding-top:.72rem">', unsafe_allow_html=True)
            if st.button(icon, key="dark_toggle", help="Dark / light mode"):
                st.session_state["dark_mode"] = not dark_mode
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        dark_mode = st.session_state.get("dark_mode", False)
        theme = DARK if dark_mode else LIGHT
        st.markdown(_sidebar_extra_css(theme), unsafe_allow_html=True)

        st.divider()

        st.markdown('<span class="sb-section-title">Initialiser l\'analyse client</span>', unsafe_allow_html=True)
        st.markdown('<span class="sb-section-sub">Lancez la démo intégrée ou importez vos propres données.</span>', unsafe_allow_html=True)

        raw_df = st.session_state.get("raw_df", None)

        st.markdown('<div class="launch-btn-wrap">', unsafe_allow_html=True)
        if st.button("\u25b6  Lancer une simulation business", use_container_width=True, key="btn_demo", type="primary"):
            try:
                raw_df = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name=1)
                st.session_state["raw_df"] = raw_df
                st.session_state["pipeline_key"] = None
                st.success(f"\u2713 {len(raw_df):,} clients charg\u00e9s")
            except Exception:
                from utils.synthetic import generate_synthetic_data
                raw_df = generate_synthetic_data(900)
                st.session_state["raw_df"] = raw_df
                st.session_state["pipeline_key"] = None
                st.info("Fichier absent \u2014 dataset synth\u00e9tique g\u00e9n\u00e9r\u00e9.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="demo-badge">'
            '<span class="demo-dot"></span> Cas business CRM/e-commerce int\u00e9gr\u00e9'
            '<span class="demo-dot"></span> 900 clients'
            '</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:.75rem 0 .6rem">'
            f'<div style="flex:1;height:1px;background:{theme["border"]};opacity:.6"></div>'
            f'<span style="font-size:.63rem;font-family:\'DM Mono\',monospace;letter-spacing:.1em;opacity:.38;text-transform:uppercase">ou</span>'
            f'<div style="flex:1;height:1px;background:{theme["border"]};opacity:.6"></div></div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:.8rem;font-weight:800;color:#1E40AF;margin-bottom:.3rem">'
            f'Importez vos donn\u00e9es clients/CRM</div>',
            unsafe_allow_html=True)

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
                _df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.session_state["uploaded_df"] = _df
                st.session_state["uploaded_name"] = uploaded_file.name
            except Exception as e:
                st.error(f"Erreur lecture : {e}")
                st.session_state.pop("uploaded_df", None)

        if st.session_state.get("uploaded_df") is not None:
            _n = len(st.session_state["uploaded_df"])
            _name = st.session_state.get("uploaded_name", "fichier")
            st.markdown(
                f'<div style="font-size:.69rem;color:{theme["protect_text"]};font-family:\'DM Mono\',monospace;padding:.2rem .1rem;margin-top:.1rem">'
                f'\u2713 {_name} \u2014 {_n:,} lignes</div>', unsafe_allow_html=True)
            st.markdown('<div class="launch-upload-btn-wrap">', unsafe_allow_html=True)
            if st.button("\u25b6  Lancer l\u2019analyse pr\u00e9dictive", use_container_width=True, key="btn_launch_upload", type="primary"):
                raw_df = st.session_state["uploaded_df"]
                st.session_state["raw_df"] = raw_df
                st.session_state["pipeline_key"] = None
                st.session_state.pop("uploaded_df", None)
                st.success(f"\u2713 {len(raw_df):,} clients import\u00e9s")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="sb-ent-wrap">'
            '<input type="checkbox" class="sb-ent-toggle" id="sb_ent_acc">'
            '<label class="sb-ent-btn" for="sb_ent_acc">'
            '<span class="sb-ent-label">Param\u00e8tres entreprises</span>'
            '<span class="sb-ent-chevron">&#9660;</span>'
            '</label>'
            '<div class="sb-ent-body">'
            '<p class="sb-ent-desc">Donn&eacute;es diff&eacute;rentes de la structure standard\u00a0? ChurnIQ peut &ecirc;tre adapt&eacute; &agrave; votre organisation.</p>'
            '<ul class="sb-ent-list">'
            '<li>Vos donn&eacute;es CRM</li>'
            '<li>Vos indicateurs business</li>'
            '<li>Votre segmentation client</li>'
            '<li>Vos r&egrave;gles de scoring</li>'
            '<li>Vos objectifs m&eacute;tier</li>'
            '</ul>'
            '<p class="sb-ent-cta">ChurnIQ n\'est qu\'un exemple de ce qu\'il est possible de construire. '
            'Contactez-nous pour cr&eacute;er des solutions intelligentes adapt&eacute;es &agrave; vos propres donn&eacute;es et enjeux m&eacute;tier.</p>'
            '<a class="sb-ent-email" href="mailto:zaravitamds18@gmail.com">&#128233; zaravitamds18@gmail.com</a>'
            '</div></div>',
            unsafe_allow_html=True)

        st.divider()

        # ── Bouton Guide ──────────────────────────────────────────────────────
        # st.button normal. La couleur #708090 est gérée dans styles.py via
        # section[data-testid="stSidebar"] [data-testid="stButton"]:last-child button
        # qui cible le dernier bouton de la sidebar (toujours le Guide).
        if st.button(
            "📖  Guide d’intégration des données",
            use_container_width=True,
            key="btn_guide"
        ):
            st.session_state["show_data_guide"] = True

        if raw_df is not None:
            st.markdown(
                '<div class="status-active"><div class="status-dot"></div>'
                f'<div>Dataset actif \u2014 <strong>{len(raw_df):,} clients</strong></div></div>',
                unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '<div style="font-size:.62rem;opacity:.28;font-family:\'DM Mono\',monospace;line-height:1.9">'
            'ZARA VITA<br>Smart Automation Technologies<br>zaravitamds18@gmail.com</div>',
            unsafe_allow_html=True)

    return raw_df, theme, st.session_state.get("show_data_guide", False)