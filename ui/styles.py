"""
ui/styles.py — Generates full CSS string from a theme dict.
"""

def build_css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: {t['bg_main']} !important;
    color: {t['text_primary']} !important;
    transition: background-color 0.3s, color 0.3s;
}}
.stApp {{ background-color: {t['bg_main']} !important; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding: 1.8rem 2.5rem 3rem; max-width: 1380px; }}
section[data-testid="stSidebar"] {{ background-color: {t['bg_sidebar']} !important; border-right: 1px solid {t['border']}; }}

/* ── Tab navigation ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {t['bg_card2']};
    border-radius: 14px;
    padding: 5px;
    border: 1px solid {t['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: {t['text_secondary']};
    background: transparent;
    border: none;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: {t['tab_active_bg']} !important;
    color: {t['accent_blue']} !important;
    border: 1px solid {t['tab_active_bd']} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

/* ── Hero ── */
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
    color: {t['accent_blue']};
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
    color: {t['text_primary']};
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
    color: {t['text_secondary']};
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}}

/* ── Section labels ── */
.sec-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {t['accent_blue']};
    margin-bottom: 0.3rem;
}}
.sec-title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {t['text_primary']};
    margin-bottom: 1.2rem;
}}
.sec-divider {{
    border: none;
    border-top: 1px solid {t['border']};
    margin: 2rem 0;
}}

/* ── KPI cards ── */
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
    color: {t['text_muted']};
    margin-bottom: 0.6rem;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {t['text_primary']};
    line-height: 1;
    letter-spacing: -0.02em;
}}
.kpi-delta {{
    font-size: 0.75rem;
    color: {t['text_muted']};
    margin-top: 0.5rem;
    font-family: 'DM Mono', monospace;
}}

/* ── Matrix segment cards ── */
.mc-priority {{ background:{t['priority_bg']}; border:1px solid {t['priority_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-protect  {{ background:{t['protect_bg']};  border:1px solid {t['protect_bd']};  border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-optimize {{ background:{t['optimize_bg']}; border:1px solid {t['optimize_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-automate {{ background:{t['automate_bg']}; border:1px solid {t['automate_bd']}; border-radius:16px; padding:1.4rem; text-align:center; }}
.mc-label    {{ font-size:0.72rem; font-family:'DM Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.7rem; }}
.mc-count    {{ font-size:2.4rem; font-weight:800; line-height:1; letter-spacing:-0.02em; }}
.mc-revenue  {{ font-size:0.78rem; color:{t['text_muted']}; margin-top:0.35rem; }}
.mc-action   {{ font-size:0.76rem; margin-top:0.9rem; padding:0.45rem 0.7rem; border-radius:9px; background:rgba(0,0,0,0.05); color:{t['text_secondary']}; }}

/* ── Insight cards ── */
.insight-card {{
    background: {t['accent_blue_bg']};
    border: 1px solid {t['accent_blue_bd']};
    border-left: 3px solid {t['accent_blue']};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}}
.insight-stat {{ font-size:1.8rem; font-weight:800; color:{t['accent_blue']}; display:block; margin-bottom:0.25rem; letter-spacing:-0.02em; }}
.insight-text {{ font-size:0.9rem; color:{t['text_secondary']}; line-height:1.6; }}

/* ── Simulation box ── */
.sim-box {{
    background: {t['protect_bg']};
    border: 1px solid {t['protect_bd']};
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}}
.sim-box h4 {{ color:{t['protect_text']}; font-size:1rem; margin-bottom:0.5rem; }}
.sim-box p  {{ color:{t['text_secondary']}; font-size:0.9rem; line-height:1.6; }}

/* ── About / Data cards ── */
.about-card {{
    background: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 1.4rem;
    height: 100%;
}}
.about-card h4 {{ font-size:0.95rem; font-weight:700; color:{t['text_primary']}; margin-bottom:0.5rem; }}
.about-card p  {{ font-size:0.82rem; color:{t['text_secondary']}; line-height:1.65; }}

/* ── Data columns table ── */
.col-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
.col-table th {{ background:{t['bg_card2']}; color:{t['text_muted']}; font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:left; border-bottom:1px solid {t['border']}; }}
.col-table td {{ padding:0.55rem 0.8rem; color:{t['text_secondary']}; border-bottom:1px solid {t['border']}; vertical-align:top; }}
.col-table tr:last-child td {{ border-bottom:none; }}
.col-table .col-name {{ font-family:'DM Mono',monospace; color:{t['accent_blue']}; font-size:0.78rem; white-space:nowrap; }}
.col-table .col-type {{ color:{t['text_muted']}; font-size:0.72rem; }}

/* ── Demo toggle ── */
.demo-label {{ font-size:0.75rem; color:{t['text_muted']}; text-align:center; font-family:'DM Mono',monospace; letter-spacing:0.06em; margin-top:0.4rem; }}

/* ── Footer ── */
.footer {{ text-align:center; padding:2rem 0 0.5rem; color:{t['text_muted']}; font-size:0.72rem; font-family:'DM Mono',monospace; letter-spacing:0.08em; border-top:1px solid {t['border']}; margin-top:2rem; }}
.footer a {{ color:{t['accent_blue']}; text-decoration:none; }}

/* ── Streamlit widget overrides ── */
.stButton > button {{
    background: {t['accent_blue']};
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.88; }}
div[data-testid="stMetric"] {{ background:{t['bg_card']}; border:1px solid {t['border']}; border-radius:14px; padding:1rem 1.2rem; }}
div[data-testid="stMetricValue"] {{ color:{t['text_primary']} !important; font-weight:800 !important; }}
div[data-testid="stMetricLabel"] {{ color:{t['text_muted']} !important; font-size:0.78rem !important; }}

/* ── Dataframe ── */
.stDataFrame {{ border-radius:14px; overflow:hidden; border:1px solid {t['border']}; }}
iframe[title="st_aggrid.agGrid"] {{ border-radius:14px; }}
</style>
"""
