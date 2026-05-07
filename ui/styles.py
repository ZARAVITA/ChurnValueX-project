"""
ui/styles.py — Generates full CSS string from a theme dict.
Covers all Streamlit native elements for both light and dark mode.
"""

def build_css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ════════════════════════════════════════════════════
   BASE — global reset
════════════════════════════════════════════════════ */
html, body {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: {t['bg_main']} !important;
    color: {t['text_primary']} !important;
}}
.stApp {{
    background-color: {t['bg_main']} !important;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding: 1.8rem 2.5rem 3rem; max-width: 1380px; }}

/* ════════════════════════════════════════════════════
   SIDEBAR — full override (all text, widgets, labels)
════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background-color: {t['bg_sidebar']} !important;
    border-right: 1px solid {t['border']} !important;
}}
/* Every text node inside sidebar */
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {{
    color: {t['text_primary']} !important;
}}
/* Muted / secondary text in sidebar */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {{
    color: {t['text_secondary']} !important;
}}
/* Sidebar toggle label */
section[data-testid="stSidebar"] [data-testid="stToggle"] span {{
    color: {t['text_primary']} !important;
}}
/* Sidebar radio / selectbox labels */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {{
    color: {t['text_secondary']} !important;
}}
/* Expander header */
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] details summary span,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary *  {{
    color: {t['text_primary']} !important;
}}
/* Expander body */
section[data-testid="stSidebar"] details[open] > div *,
section[data-testid="stSidebar"] [data-testid="stExpander"] > div * {{
    color: {t['text_secondary']} !important;
    background-color: transparent !important;
}}
/* Expander container background */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: {t['bg_card2']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
}}
/* Divider */
section[data-testid="stSidebar"] hr {{
    border-color: {t['border']} !important;
}}
/* Sidebar button */
section[data-testid="stSidebar"] .stButton > button {{
    background: {t['accent_blue']} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
/* Success / info banners in sidebar */
section[data-testid="stSidebar"] [data-testid="stAlert"] *,
section[data-testid="stSidebar"] .stSuccess *,
section[data-testid="stSidebar"] .stInfo * {{
    color: {t['text_primary']} !important;
}}

/* ════════════════════════════════════════════════════
   MAIN CONTENT — global text nodes
════════════════════════════════════════════════════ */
/* Markdown text */
.stMarkdown p, .stMarkdown li, .stMarkdown span {{
    color: {t['text_secondary']} !important;
}}
/* Bold inside markdown */
.stMarkdown strong {{
    color: {t['text_primary']} !important;
}}
/* All Streamlit-generated text labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"] {{
    color: {t['text_secondary']} !important;
    font-size: 0.85rem !important;
}}
/* Headings */
h1, h2, h3, h4 {{
    color: {t['text_primary']} !important;
}}
/* Input / select / number_input boxes */
input, select, textarea {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
}}
/* Multiselect tags */
[data-baseweb="tag"] {{
    background-color: {t['accent_blue_bg']} !important;
    color: {t['accent_blue']} !important;
}}
[data-baseweb="tag"] span {{
    color: {t['accent_blue']} !important;
}}
/* Slider track and thumb labels */
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {{
    color: {t['text_secondary']} !important;
}}
/* Selectbox dropdown */
[data-baseweb="select"] div {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
    border-color: {t['border']} !important;
}}
[data-baseweb="popover"] *,
[data-baseweb="menu"] * {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
}}
/* Number input */
[data-testid="stNumberInput"] input {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
    border-color: {t['border']} !important;
}}

/* ════════════════════════════════════════════════════
   TABS  ← FIXED: clear separations between tabs
════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {t['bg_card2']};
    border-radius: 14px;
    padding: 5px 6px;
    border: 1px solid {t['border_strong']};
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: {t['text_secondary']} !important;
    background: transparent;
    border: 1px solid transparent !important;
    transition: all 0.18s ease;
    position: relative;
}}
/* Separator pipe between inactive tabs */
.stTabs [data-baseweb="tab"]:not(:last-child)::after {{
    content: '';
    position: absolute;
    right: -3px;
    top: 22%;
    height: 56%;
    width: 1px;
    background: {t['border_strong']};
    border-radius: 1px;
    opacity: 0.7;
    pointer-events: none;
}}
/* Hide separator adjacent to the active tab */
.stTabs [aria-selected="true"]::after {{
    opacity: 0 !important;
}}
/* Hover state for inactive tabs */
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    background: {t['bg_card']} !important;
    border-color: {t['border']} !important;
    color: {t['text_primary']} !important;
}}
/* Active tab */
.stTabs [aria-selected="true"] {{
    background: {t['tab_active_bg']} !important;
    color: {t['accent_blue']} !important;
    border: 1px solid {t['tab_active_bd']} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
}}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

/* ════════════════════════════════════════════════════
   HERO
════════════════════════════════════════════════════ */
.hero-wrap {{
    background: linear-gradient(135deg, {t['bg_card']} 0%, {t['bg_card2']} 100%);
    border: 1px solid {t['border']};
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}
.hero-wrap::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, {t['accent_blue_bg']} 0%, transparent 70%);
    pointer-events: none;
}}
.hero-badge {{
    display: inline-block;
    background: {t['accent_blue_bg']};
    border: 1px solid {t['accent_blue_bd']};
    color: {t['accent_blue']} !important;
    padding: 0.25rem 0.9rem;
    border-radius: 100px;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: {t['text_primary']} !important;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}}
.hero-title span {{
    background: linear-gradient(120deg, {t['hero_grad_a']}, {t['hero_grad_b']}, {t['hero_grad_c']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero-subtitle {{
    font-size: 1rem;
    color: {t['text_secondary']} !important;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}}

/* ════════════════════════════════════════════════════
   SECTION LABELS
════════════════════════════════════════════════════ */
.sec-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {t['accent_blue']} !important;
    margin-bottom: 0.3rem;
}}
.sec-title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {t['text_primary']} !important;
    margin-bottom: 1.2rem;
}}
.sec-divider {{
    border: none;
    border-top: 1px solid {t['border']};
    margin: 2rem 0;
}}

/* ════════════════════════════════════════════════════
   KPI CARDS
════════════════════════════════════════════════════ */
.kpi-card {{
    background: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {t['accent_blue']}, transparent);
    border-radius: 16px 16px 0 0;
}}
.kpi-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {t['text_muted']} !important;
    margin-bottom: 0.6rem;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {t['text_primary']} !important;
    line-height: 1;
    letter-spacing: -0.02em;
}}
.kpi-delta {{
    font-size: 0.75rem;
    color: {t['text_muted']} !important;
    margin-top: 0.5rem;
    font-family: 'DM Mono', monospace;
}}

/* ════════════════════════════════════════════════════
   SEGMENT MATRIX CARDS
════════════════════════════════════════════════════ */
.mc-priority {{ background:{t['priority_bg']}; border:1px solid {t['priority_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-protect  {{ background:{t['protect_bg']};  border:1px solid {t['protect_bd']};  border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-optimize {{ background:{t['optimize_bg']}; border:1px solid {t['optimize_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-automate {{ background:{t['automate_bg']}; border:1px solid {t['automate_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-label    {{ font-size:0.72rem; font-family:'DM Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.7rem; }}
.mc-count    {{ font-size:2.4rem; font-weight:800; line-height:1; letter-spacing:-0.02em; }}
.mc-revenue  {{ font-size:0.78rem; color:{t['text_muted']} !important; margin-top:0.35rem; }}
.mc-action   {{ font-size:0.76rem; margin-top:0.9rem; padding:0.45rem 0.7rem; border-radius:9px;
                background:{t['bg_card2']}; color:{t['text_secondary']} !important; }}

/* ════════════════════════════════════════════════════
   INSIGHT CARDS
════════════════════════════════════════════════════ */
.insight-card {{
    background: {t['accent_blue_bg']};
    border: 1px solid {t['accent_blue_bd']};
    border-left: 3px solid {t['accent_blue']};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}}
.insight-stat {{ font-size:1.8rem; font-weight:800; color:{t['accent_blue']} !important; display:block; margin-bottom:0.25rem; letter-spacing:-0.02em; }}
.insight-text {{ font-size:0.9rem; color:{t['text_secondary']} !important; line-height:1.6; }}

/* ════════════════════════════════════════════════════
   SIMULATION BOX
════════════════════════════════════════════════════ */
.sim-box {{
    background: {t['protect_bg']};
    border: 1px solid {t['protect_bd']};
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}}
.sim-box h4 {{ color:{t['protect_text']} !important; font-size:1rem; margin-bottom:0.5rem; }}
.sim-box p  {{ color:{t['text_secondary']} !important; font-size:0.9rem; line-height:1.6; }}

/* ════════════════════════════════════════════════════
   ABOUT / DATA CARDS
════════════════════════════════════════════════════ */
.about-card {{
    background: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 1.4rem;
    height: 100%;
}}
.about-card h4 {{ font-size:0.95rem; font-weight:700; color:{t['text_primary']} !important; margin-bottom:0.5rem; }}
.about-card p  {{ font-size:0.82rem; color:{t['text_secondary']} !important; line-height:1.65; }}

/* ════════════════════════════════════════════════════
   DATA TABLE (sidebar column docs)
════════════════════════════════════════════════════ */
.col-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
.col-table th {{
    background:{t['bg_card2']};
    color:{t['text_muted']} !important;
    font-family:'DM Mono',monospace;
    font-size:0.62rem;
    letter-spacing:0.1em;
    text-transform:uppercase;
    padding:0.6rem 0.8rem;
    text-align:left;
    border-bottom:1px solid {t['border']};
}}
.col-table td {{
    padding:0.55rem 0.8rem;
    color:{t['text_secondary']} !important;
    border-bottom:1px solid {t['border']};
    vertical-align:top;
    background: transparent !important;
}}
.col-table tr:last-child td {{ border-bottom:none; }}
.col-table .col-name {{ font-family:'DM Mono',monospace; color:{t['accent_blue']} !important; font-size:0.78rem; white-space:nowrap; }}
.col-table .col-type {{ color:{t['text_muted']} !important; font-size:0.72rem; }}

/* ════════════════════════════════════════════════════
   DEMO LABEL / FOOTER
════════════════════════════════════════════════════ */
.demo-label {{
    font-size:0.75rem;
    color:{t['text_muted']} !important;
    text-align:center;
    font-family:'DM Mono',monospace;
    letter-spacing:0.06em;
    margin-top:0.4rem;
}}
.footer {{
    text-align:center;
    padding:2rem 0 0.5rem;
    color:{t['text_muted']} !important;
    font-size:0.72rem;
    font-family:'DM Mono',monospace;
    letter-spacing:0.08em;
    border-top:1px solid {t['border']};
    margin-top:2rem;
}}
.footer a {{ color:{t['accent_blue']} !important; text-decoration:none; }}

/* ════════════════════════════════════════════════════
   STREAMLIT NATIVE WIDGET OVERRIDES
════════════════════════════════════════════════════ */
/* Primary buttons */
.stButton > button {{
    background: {t['accent_blue']} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

/* Download buttons */
[data-testid="stDownloadButton"] > button {{
    background: {t['bg_card2']} !important;
    color: {t['text_primary']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}

/* Metric widgets */
div[data-testid="stMetric"] {{
    background:{t['bg_card']} !important;
    border:1px solid {t['border']} !important;
    border-radius:14px !important;
    padding:1rem 1.2rem !important;
}}
div[data-testid="stMetricValue"] {{
    color:{t['text_primary']} !important;
    font-weight:800 !important;
}}
div[data-testid="stMetricLabel"] {{
    color:{t['text_muted']} !important;
    font-size:0.78rem !important;
}}

/* Alert / info / success boxes */
[data-testid="stAlert"] {{
    background: {t['bg_card2']} !important;
    border-color: {t['border']} !important;
}}
[data-testid="stAlert"] * {{
    color: {t['text_primary']} !important;
}}

/* Expander in main content */
[data-testid="stExpander"] {{
    background: {t['bg_card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary * {{
    color: {t['text_primary']} !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] > div * {{
    color: {t['text_secondary']} !important;
}}

/* Dataframe */
.stDataFrame {{
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid {t['border']} !important;
}}

/* Spinner text */
[data-testid="stSpinner"] * {{
    color: {t['text_secondary']} !important;
}}

/* Multiselect box */
[data-baseweb="select"] > div {{
    background-color: {t['bg_card']} !important;
    border-color: {t['border']} !important;
}}
[data-baseweb="select"] span {{
    color: {t['text_primary']} !important;
}}

</style>
"""