# -*- coding: utf-8 -*-
"""
Customer Churn Intelligence System
Premium SaaS-grade Streamlit application
Developed by ZARA VITA — Smart Automation Technologies
"""

import os
os.environ["STREAMLIT_USE_ARROW"] = "0"

import streamlit as st

st.set_page_config(
    page_title="ChurnIQ — Customer Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⬡"
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080C14;
    color: #E2E8F0;
}
.stApp { background-color: #080C14; }

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Hero */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #F8FAFC 0%, #64B5F6 50%, #42A5F5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #64748B;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(66,165,245,0.12);
    border: 1px solid rgba(66,165,245,0.3);
    color: #42A5F5;
    padding: 0.25rem 0.8rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2.5rem 0;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #42A5F5;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 1.5rem;
}

/* Metric cards */
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #42A5F5, transparent);
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.4rem;
}

/* Segment tags */
.tag-priority { background:#EF444420; color:#EF4444; border:1px solid #EF444450; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.tag-protect  { background:#22C55E20; color:#22C55E; border:1px solid #22C55E50; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.tag-optimize { background:#F59E0B20; color:#F59E0B; border:1px solid #F59E0B50; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.tag-automate { background:#64748B20; color:#94A3B8; border:1px solid #64748B50; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/* Insight card */
.insight-card {
    background: rgba(66,165,245,0.06);
    border: 1px solid rgba(66,165,245,0.2);
    border-left: 3px solid #42A5F5;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.insight-text { font-size: 0.95rem; color: #CBD5E1; line-height: 1.6; }
.insight-stat { font-size: 1.5rem; font-weight: 800; color: #42A5F5; display:block; margin-bottom:0.3rem; }

/* Segment matrix card */
.matrix-card {
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
}
.matrix-card h3 { font-size: 0.8rem; font-family:'Space Mono',monospace; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.8rem; }
.matrix-count { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.matrix-revenue { font-size: 0.8rem; color: #94A3B8; margin-top: 0.3rem; }
.matrix-action { font-size: 0.78rem; margin-top: 0.8rem; padding: 0.4rem 0.8rem; border-radius: 8px; background: rgba(255,255,255,0.07); }

/* About section */
.about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.about-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem;
}
.about-card h4 { font-size: 0.9rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.5rem; }
.about-card p  { font-size: 0.8rem; color: #64748B; line-height: 1.6; }

/* Simulation box */
.sim-box {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.sim-box h4 { color: #22C55E; font-size: 1rem; margin-bottom: 0.5rem; }
.sim-box p  { color: #CBD5E1; font-size: 0.9rem; line-height: 1.6; }

/* Stacked bar for demo toggle */
.demo-label { font-size: 0.78rem; color: #64748B; text-align:center; font-family:'Space Mono',monospace; letter-spacing:0.05em; margin-top: 0.4rem; }

/* Dataframe overrides */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from lifelines import WeibullAFTFitter
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

SEED = 42
np.random.seed(SEED)

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", color="#94A3B8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
)

# ─── Data utilities ───────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 800) -> pd.DataFrame:
    """Realistic synthetic e-commerce dataset."""
    np.random.seed(SEED)
    df = pd.DataFrame({
        "CustomerID": range(10000, 10000 + n),
        "Churn": np.random.choice([0, 1], size=n, p=[0.83, 0.17]),
        "Tenure": np.random.gamma(3, 5, n).clip(1, 60).round(1),
        "CityTier": np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
        "WarehouseToHome": np.random.randint(5, 40, n),
        "HourSpendOnApp": np.random.gamma(2, 1.5, n).clip(0, 8).round(1),
        "NumberOfDeviceRegistered": np.random.randint(1, 6, n),
        "SatisfactionScore": np.random.randint(1, 6, n),
        "NumberOfAddress": np.random.randint(1, 10, n),
        "Complain": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "OrderAmountHikeFromlastYear": np.random.normal(15, 8, n).clip(0, 50).round(1),
        "CouponUsed": np.random.randint(0, 15, n),
        "OrderCount": np.random.randint(1, 20, n),
        "DaySinceLastOrder": np.random.randint(0, 30, n),
        "CashbackAmount": np.random.gamma(4, 50, n).round(2),
        "Gender": np.random.choice(["Male", "Female"], n),
        "PreferredLoginDevice": np.random.choice(["Mobile Phone", "Computer", "Phone"], n, p=[0.6, 0.2, 0.2]),
        "PreferredPaymentMode": np.random.choice(["Debit Card", "Credit Card", "UPI", "E wallet", "Cash on Delivery"], n, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        "PreferedOrderCat": np.random.choice(["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"], n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        "MaritalStatus": np.random.choice(["Married", "Single", "Divorced"], n, p=[0.55, 0.33, 0.12]),
    })
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()

# ─── ML pipeline ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame):
    df = _clean(df_raw.copy())
    df = _segment(df)
    df, model, features = _predict_churn(df)
    df = _predict_survival(df)
    df, clv_df = _predict_clv(df_raw.copy(), df)
    df = df.merge(clv_df, on="CustomerID", how="left")
    df["CLV_12mois"] = df["CLV_12mois"].fillna(0)
    df["SPA_score"] = df["Churn_proba"] * df["CLV_12mois"]
    df = _assign_matrix_segment(df)
    return df

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if "Tenure" in df.columns:
        df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median()).replace(0, 0.001)
    if "DaySinceLastOrder" in df.columns:
        df["DaySinceLastOrder"] = df["DaySinceLastOrder"].fillna(df["DaySinceLastOrder"].max())
    if "HourSpendOnApp" in df.columns:
        df["HourSpendOnApp"] = df["HourSpendOnApp"].fillna(df["HourSpendOnApp"].median())
    if "CashbackAmount" in df.columns:
        df["CashbackAmount"] = df["CashbackAmount"].fillna(df["CashbackAmount"].median())
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"].astype(str))
    cats = [c for c in ["PreferredLoginDevice","PreferredPaymentMode","PreferedOrderCat","MaritalStatus"] if c in df.columns]
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=True)
    df[df.select_dtypes(include=["bool"]).columns] = df.select_dtypes(include=["bool"]).astype(int)
    return df

def _segment(df: pd.DataFrame, n_clusters: int = 2) -> pd.DataFrame:
    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
    sc = StandardScaler()
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    df["segment"] = km.fit_predict(sc.fit_transform(X))
    return df

def _predict_churn(df: pd.DataFrame):
    features = [c for c in df.columns if c not in ["CustomerID", "Churn"]]
    X, y = df[features], df.get("Churn", pd.Series(np.zeros(len(df)), index=df.index))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y if y.nunique() > 1 else None, random_state=SEED)
    if y.nunique() > 1:
        sm = SMOTE(random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=SEED, n_estimators=70)
    model.fit(X_tr, y_tr)
    df["Churn_proba"] = model.predict_proba(df[features])[:, 1]
    df["Churn_pred"] = model.predict(df[features])
    return df, model, features

def _predict_survival(df: pd.DataFrame) -> pd.DataFrame:
    dur, evt = "Tenure", "Churn"
    df[dur] = df[dur].replace(0, 0.001)
    covs = [c for c in [
        "CityTier","WarehouseToHome","Gender","HourSpendOnApp","NumberOfDeviceRegistered",
        "SatisfactionScore","NumberOfAddress","Complain","OrderAmountHikeFromlastYear","CouponUsed",
        "OrderCount","DaySinceLastOrder","CashbackAmount","PreferredLoginDevice_Mobile Phone",
        "PreferredLoginDevice_Phone","PreferredPaymentMode_COD","PreferredPaymentMode_Cash on Delivery",
        "PreferredPaymentMode_Credit Card","PreferredPaymentMode_Debit Card","PreferredPaymentMode_E wallet",
        "PreferredPaymentMode_UPI","PreferedOrderCat_Grocery","PreferedOrderCat_Laptop & Accessory",
        "PreferedOrderCat_Mobile","PreferedOrderCat_Mobile Phone","PreferedOrderCat_Others",
        "MaritalStatus_Single","segment"
    ] if c in df.columns]
    aft = WeibullAFTFitter()
    aft.fit(df[[dur, evt] + covs].dropna(), duration_col=dur, event_col=evt)
    h1 = df[dur] + 1
    h3 = df[dur] + 3
    s0 = aft.predict_survival_function(df[covs], times=df[dur]).T
    s1 = aft.predict_survival_function(df[covs], times=h1).T
    s3 = aft.predict_survival_function(df[covs], times=h3).T
    df["proba_churn_1mois"] = (s0.values[:, 0] - s1.values[:, 0]).clip(0, 1)
    df["proba_churn_3mois"] = (s0.values[:, 0] - s3.values[:, 0]).clip(0, 1)
    s_now = aft.predict_survival_function(df[covs], times=df[dur]).T
    df["proba_churn_now"] = (1 - s_now.values[:, 0]).clip(0, 1)
    return df

def _predict_clv(df_raw: pd.DataFrame, df_proc: pd.DataFrame):
    """BG/NBD + Gamma-Gamma CLV — returns merged CLV column."""
    df = df_raw.copy()
    # ── Harden all numeric columns used below before any computation ──
    for col in ["Tenure", "OrderCount", "CashbackAmount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median()).replace(0, 0.001)
    if "OrderCount" in df.columns:
        df["OrderCount"] = df["OrderCount"].fillna(0).clip(lower=0)
    if "CashbackAmount" in df.columns:
        df["CashbackAmount"] = df["CashbackAmount"].fillna(df["CashbackAmount"].median())

    df["Tenure_days"] = (df["Tenure"] * 30).round().fillna(0).astype(int)
    df_c = df[df["Tenure_days"] >= 50].copy()
    df_c["Tenure_months"] = (df_c["Tenure_days"] / 30).round().fillna(1).astype(int)
    analysis_date = pd.Timestamp("2025-11-18")
    mean_ob = df_c.groupby("Tenure_months")["OrderCount"].mean().to_dict() if "OrderCount" in df_c.columns else {}
    txns = []
    for _, row in df_c.iterrows():
        tenure_m = int(row["Tenure_months"]) if pd.notna(row["Tenure_months"]) else 1
        order_c  = float(row["OrderCount"]) if ("OrderCount" in row and pd.notna(row["OrderCount"])) else 1.0
        n = max(1, int(tenure_m * max(order_c, 1)))
        tenure_d = int(row["Tenure_days"]) if pd.notna(row["Tenure_days"]) else 30
        start = analysis_date - pd.to_timedelta(tenure_d, unit="D")
        dates = pd.date_range(start=start, end=analysis_date, periods=n) + pd.to_timedelta(np.random.randint(-7, 7, n), unit="D")
        cr = 0.05
        cashback = float(row["CashbackAmount"]) if ("CashbackAmount" in row and pd.notna(row["CashbackAmount"])) else 5.0
        mean_amt = (cashback / cr) / max(1.0, order_c)
        amounts = np.random.gamma(2, max(mean_amt / 2, 0.01), n)
        for d, a in zip(dates, amounts):
            txns.append([row["CustomerID"], d, max(0.01, a)])
    df_txn = pd.DataFrame(txns, columns=["CustomerID", "Date", "Amount"])
    if df_txn.empty:
        return df_proc, pd.DataFrame(columns=["CustomerID", "CLV_12mois"])
    summary = summary_data_from_transaction_data(df_txn, "CustomerID", "Date", "Amount", observation_period_end=analysis_date)
    summary = summary[summary["monetary_value"] > 0]
    sumg = summary[summary["frequency"] > 0].copy()
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(sumg["frequency"], sumg["monetary_value"])
    clv = ggf.customer_lifetime_value(bgf, summary["frequency"], summary["recency"], summary["T"], summary["monetary_value"], time=12, freq="D", discount_rate=0.01)
    clv = clv.reset_index()
    clv.columns = ["CustomerID", "CLV_12mois"]
    clv["CLV_12mois"] = (clv["CLV_12mois"] / 10).clip(lower=0)
    return df_proc, clv

def _assign_matrix_segment(df: pd.DataFrame) -> pd.DataFrame:
    clv_med = df["CLV_12mois"].median()
    risk_thresh = 0.5
    conds = [
        (df["Churn_proba"] >= risk_thresh) & (df["CLV_12mois"] >= clv_med),
        (df["Churn_proba"] < risk_thresh)  & (df["CLV_12mois"] >= clv_med),
        (df["Churn_proba"] >= risk_thresh) & (df["CLV_12mois"] < clv_med),
        (df["Churn_proba"] < risk_thresh)  & (df["CLV_12mois"] < clv_med),
    ]
    choices = ["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"]
    df["Matrix_Segment"] = np.select(conds, choices, default="AUTOMATE")
    return df

# ─── Plot helpers ─────────────────────────────────────────────────────────────

SEG_COLORS = {"PRIORITY": "#EF4444", "PROTECT": "#22C55E", "OPTIMIZE": "#F59E0B", "AUTOMATE": "#64748B"}

def fig_decision_matrix(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="CLV_12mois", y="Churn_proba",
        color="Matrix_Segment",
        color_discrete_map=SEG_COLORS,
        hover_data={"CustomerID": True, "CLV_12mois": ":.1f", "Churn_proba": ":.0%", "Matrix_Segment": True},
        labels={"CLV_12mois": "Customer Lifetime Value (12M)", "Churn_proba": "Churn Risk"},
        opacity=0.75,
        template=PLOTLY_TEMPLATE,
        size_max=8,
    )
    # quadrant lines
    clv_med = df["CLV_12mois"].median()
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.add_vline(x=clv_med, line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickformat=".0%",
    )
    return fig

def fig_demo_toggle(df: pd.DataFrame, with_system: bool) -> go.Figure:
    if not with_system:
        fig = px.scatter(
            df, x="CLV_12mois", y="Churn_proba",
            color_discrete_sequence=["#475569"],
            template=PLOTLY_TEMPLATE,
            labels={"CLV_12mois": "CLV", "Churn_proba": "Churn Risk"},
        )
        fig.update_traces(marker=dict(size=6, opacity=0.5))
        fig.update_layout(height=340, title=dict(text="No segmentation — raw data", font=dict(size=13, color="#64748B")))
    else:
        fig = px.scatter(
            df, x="CLV_12mois", y="Churn_proba",
            color="Matrix_Segment",
            color_discrete_map=SEG_COLORS,
            template=PLOTLY_TEMPLATE,
            labels={"CLV_12mois": "CLV", "Churn_proba": "Churn Risk"},
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        clv_med = df["CLV_12mois"].median()
        fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
        fig.add_vline(x=clv_med, line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
        fig.update_layout(height=340, title=dict(text="Structured segmentation — 4 action zones", font=dict(size=13, color="#94A3B8")))
    return fig

def fig_clv_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="CLV_12mois", nbins=50, template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=["#42A5F5"])
    fig.update_layout(height=260, showlegend=False,
                      xaxis_title="CLV (12 months)", yaxis_title="Customers")
    return fig

def fig_risk_dist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Churn_proba", nbins=50, template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=["#F59E0B"])
    fig.update_layout(height=260, showlegend=False,
                      xaxis_title="Churn Probability", yaxis_title="Customers",
                      xaxis_tickformat=".0%")
    return fig

# ─── APP ──────────────────────────────────────────────────────────────────────

# ── Hero ──
st.markdown('<div class="hero-badge">⬡ Enterprise Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Customer Churn<br>Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Prioritize customers. Protect revenue. Act with precision.</div>', unsafe_allow_html=True)

# ── Data source ──
col_data, col_spacer = st.columns([2, 3])
with col_data:
    data_mode = st.radio(
        "Data source",
        ["Use demo dataset", "Upload your data"],
        horizontal=True,
        label_visibility="collapsed"
    )

uploaded_df = None
if data_mode == "Use demo dataset":
    if st.button("▶  Launch demo analysis", type="primary"):
        try:
            sample = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name=1)
            st.session_state["raw_df"] = sample
        except Exception:
            st.session_state["raw_df"] = generate_synthetic_data(800)
            st.info("Sample file not found — generated synthetic dataset with 800 customers.")
else:
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], label_visibility="collapsed")
    if file:
        try:
            uploaded_df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file, sheet_name=0)
            st.session_state["raw_df"] = uploaded_df
            st.success(f"File loaded — {len(uploaded_df):,} customers detected.")
        except Exception as e:
            st.error(f"File read error: {e}")

if "raw_df" not in st.session_state:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#475569">
        <div style="font-size:2rem;margin-bottom:0.5rem">⬡</div>
        <div style="font-size:1rem">Load a dataset above to activate the intelligence system</div>
        <div style="font-size:0.8rem;margin-top:0.5rem;font-family:'Space Mono',monospace">
            Accepts: CSV · XLSX · Integrated demo
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run pipeline ──
df_raw = st.session_state["raw_df"]
with st.spinner("Running analytics pipeline..."):
    df = run_pipeline(df_raw)

# ─── SECTION 1: KPIs ──────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">01 — Core Metrics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)

total = len(df)
high_risk_pct = (df["Churn_proba"] >= 0.5).mean()
clv_med = df["CLV_12mois"].median()
revenue_at_risk = df.loc[df["Churn_proba"] >= 0.5, "CLV_12mois"].sum()
top20_clv_pct = df.nlargest(int(total * 0.2), "CLV_12mois")["CLV_12mois"].sum() / df["CLV_12mois"].sum() if df["CLV_12mois"].sum() > 0 else 0
avg_clv = df["CLV_12mois"].mean()

k1, k2, k3, k4 = st.columns(4)
def kpi(col, label, value, delta=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(k1, "Customers at high risk", f"{high_risk_pct:.1%}", f"{int(high_risk_pct*total):,} of {total:,} customers")
kpi(k2, "Revenue at risk (CLV)", f"${revenue_at_risk:,.0f}", "12-month projected value")
kpi(k3, "Top-20% value concentration", f"{top20_clv_pct:.1%}", "of total CLV in top quintile")
kpi(k4, "Average CLV (12 months)", f"${avg_clv:,.1f}", "per customer")

# Distribution charts
st.markdown("<br>", unsafe_allow_html=True)
dc1, dc2 = st.columns(2)
with dc1:
    st.plotly_chart(fig_clv_dist(df), use_container_width=True, config={"displayModeBar": False})
with dc2:
    st.plotly_chart(fig_risk_dist(df), use_container_width=True, config={"displayModeBar": False})

# ─── SECTION 2: Demo toggle ───────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">02 — System Impact</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Before vs After Intelligence Layer</div>', unsafe_allow_html=True)

demo_col1, demo_col2 = st.columns(2)
with demo_col1:
    st.plotly_chart(fig_demo_toggle(df, False), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="demo-label">WITHOUT SYSTEM — No visibility, no action</div>', unsafe_allow_html=True)
with demo_col2:
    st.plotly_chart(fig_demo_toggle(df, True), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="demo-label">WITH SYSTEM — 4 clear zones, immediate action</div>', unsafe_allow_html=True)

# ─── SECTION 3: Decision matrix ──────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">03 — Strategic Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Decision Matrix</div>', unsafe_allow_html=True)

st.plotly_chart(fig_decision_matrix(df), use_container_width=True, config={"displayModeBar": False})

seg_info = {
    "PRIORITY": {"bg": "rgba(239,68,68,0.08)", "border": "#EF4444", "label": "🔴 PRIORITY", "action": "Immediate personal outreach + incentive"},
    "PROTECT":  {"bg": "rgba(34,197,94,0.08)",  "border": "#22C55E", "label": "🟢 PROTECT",  "action": "VIP loyalty program — keep them engaged"},
    "OPTIMIZE": {"bg": "rgba(245,158,11,0.08)", "border": "#F59E0B", "label": "🟡 OPTIMIZE", "action": "Targeted retention campaign — ROI check first"},
    "AUTOMATE": {"bg": "rgba(100,116,139,0.08)","border": "#64748B", "label": "⚫ AUTOMATE", "action": "Low-cost re-engagement automation"},
}

mc1, mc2, mc3, mc4 = st.columns(4)
for seg, col in zip(["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"], [mc1, mc2, mc3, mc4]):
    seg_df = df[df["Matrix_Segment"] == seg]
    rev = seg_df["CLV_12mois"].sum()
    info = seg_info[seg]
    col.markdown(f"""
    <div class="matrix-card" style="background:{info['bg']};border:1px solid {info['border']}40;">
        <h3 style="color:{info['border']}">{info['label']}</h3>
        <div class="matrix-count" style="color:{info['border']}">{len(seg_df):,}</div>
        <div class="matrix-revenue">Revenue: ${rev:,.0f}</div>
        <div class="matrix-action">{info['action']}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── SECTION 4: Customer list ─────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">04 — Customer Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Interactive Customer List</div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)
with f1:
    seg_filter = st.multiselect("Segment", ["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"], default=["PRIORITY", "PROTECT"])
with f2:
    clv_max = float(df["CLV_12mois"].quantile(0.99)) or 1.0
    clv_range = st.slider("CLV range ($)", 0.0, clv_max, (0.0, clv_max), step=1.0)
with f3:
    sort_by = st.selectbox("Sort by", ["Churn_proba", "CLV_12mois", "SPA_score", "proba_churn_1mois"])

def recommend(row):
    if row["Matrix_Segment"] == "PRIORITY":
        return "🔴 Urgent call + exclusive offer"
    if row["Matrix_Segment"] == "PROTECT":
        return "🟢 VIP program enrollment"
    if row["Matrix_Segment"] == "OPTIMIZE":
        return "🟡 Email + coupon campaign"
    return "⚫ Low-cost re-engagement"

df_view = df.copy()
df_view["Recommended Action"] = df_view.apply(recommend, axis=1)

mask = (
    df_view["Matrix_Segment"].isin(seg_filter) &
    df_view["CLV_12mois"].between(clv_range[0], clv_range[1])
)
df_view = df_view[mask].sort_values(sort_by, ascending=False)

display_cols = {
    "CustomerID": "Customer ID",
    "CLV_12mois": "CLV (12M $)",
    "Churn_proba": "Risk",
    "proba_churn_1mois": "Risk 1M",
    "proba_churn_3mois": "Risk 3M",
    "Matrix_Segment": "Segment",
    "Recommended Action": "Action",
}
df_display = df_view[list(display_cols.keys())].rename(columns=display_cols).copy()
df_display["CLV (12M $)"] = df_display["CLV (12M $)"].round(1)
df_display["Risk"] = df_display["Risk"].map(lambda x: f"{x:.1%}")
df_display["Risk 1M"] = df_display["Risk 1M"].map(lambda x: f"{x:.1%}")
df_display["Risk 3M"] = df_display["Risk 3M"].map(lambda x: f"{x:.1%}")

st.markdown(f"<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.5rem'>{len(df_display):,} customers matching filters</div>", unsafe_allow_html=True)
st.dataframe(df_display.head(200), use_container_width=True, height=380)

# ─── SECTION 5: Key insights ─────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">05 — Business Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Consulting Highlights</div>', unsafe_allow_html=True)

priority_rev = df[df["Matrix_Segment"] == "PRIORITY"]["CLV_12mois"].sum()
total_rev = df["CLV_12mois"].sum()
priority_rev_pct = priority_rev / total_rev if total_rev > 0 else 0
top20_n = int(total * 0.2)
top20_rev = df.nlargest(top20_n, "CLV_12mois")["CLV_12mois"].sum()
top20_rev_pct = top20_rev / total_rev if total_rev > 0 else 0
protect_high = (df[df["Matrix_Segment"] == "PROTECT"]["Churn_proba"] > 0.4).mean()

insights = [
    (f"{priority_rev_pct:.0%}", f"of your 12-month revenue pipeline is held by PRIORITY customers — high-value clients currently at risk of churning. These require immediate intervention."),
    (f"{top20_rev_pct:.0%}", f"of total projected revenue is concentrated in the top 20% of customers ({top20_n:,} accounts). Losing any of them has outsized financial impact."),
    (f"{int(high_risk_pct*total):,}", f"customers are currently above the 50% churn threshold. Targeting the PRIORITY segment first maximises retention ROI per marketing dollar spent."),
]

for stat, text in insights:
    st.markdown(f"""
    <div class="insight-card">
        <span class="insight-stat">{stat}</span>
        <span class="insight-text">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── BONUS: Simulation ────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">06 — Revenue Simulation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What-If Analysis</div>', unsafe_allow_html=True)

sim_col, sim_result = st.columns([2, 3])
with sim_col:
    reduction = st.slider("Churn reduction on PRIORITY customers (%)", 0, 50, 10)

priority_df = df[df["Matrix_Segment"] == "PRIORITY"].copy()
saved_customers = int(len(priority_df) * reduction / 100)
saved_revenue = priority_df.nlargest(saved_customers, "CLV_12mois")["CLV_12mois"].sum() if saved_customers > 0 else 0

with sim_result:
    st.markdown(f"""
    <div class="sim-box">
        <h4>Simulation result: −{reduction}% churn on PRIORITY segment</h4>
        <p>
            <strong style="color:#22C55E;font-size:1.2rem">${saved_revenue:,.0f}</strong> in additional 12-month revenue retained<br>
            across <strong>{saved_customers:,} customers</strong> rescued from churn.<br><br>
            This assumes your retention actions succeed proportionally on the highest-value accounts first.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── SECTION 7: About ─────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">07 — System Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">About This System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="about-grid">
    <div class="about-card">
        <h4>🎯 Churn Prediction</h4>
        <p>XGBoost classifier with SMOTE balancing. Outputs individual churn probability for each customer, used to build the risk axis of the decision matrix.</p>
    </div>
    <div class="about-card">
        <h4>⏱ Survival Analysis</h4>
        <p>Weibull AFT model estimates the probability of churn within 1 and 3 months, accounting for customer tenure and behavioral covariates.</p>
    </div>
    <div class="about-card">
        <h4>💰 CLV Estimation</h4>
        <p>BG/NBD model predicts future purchase frequency; Gamma-Gamma model estimates monetary value. Combined to produce 12-month customer lifetime value.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Downloads ────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">08 — Export</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Download Targeted Lists</div>', unsafe_allow_html=True)

dl1, dl2, dl3 = st.columns(3)
export_cols = ["CustomerID","CLV_12mois","Churn_proba","proba_churn_1mois","proba_churn_3mois","Matrix_Segment","SPA_score"]

with dl1:
    data = to_excel_bytes(df[df["Matrix_Segment"] == "PRIORITY"][export_cols].sort_values("SPA_score", ascending=False))
    st.download_button("⬇  PRIORITY customers", data=data, file_name="priority_customers.xlsx", use_container_width=True)

with dl2:
    data2 = to_excel_bytes(df.sort_values("proba_churn_1mois", ascending=False)[export_cols].head(500))
    st.download_button("⬇  At-risk next 30 days", data=data2, file_name="risk_1month.xlsx", use_container_width=True)

with dl3:
    data3 = to_excel_bytes(df.sort_values("CLV_12mois", ascending=False)[export_cols].head(500))
    st.download_button("⬇  Top CLV customers", data=data3, file_name="top_clv.xlsx", use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem;color:#334155">
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase">
        ChurnIQ — Customer Churn Intelligence System
    </div>
    <div style="font-size:0.75rem;margin-top:0.4rem">
        Developed by <strong style="color:#64748B">ZARA VITA</strong> · Smart Automation Technologies ·
        <a href="mailto:zaravitamds18@gmail.com" style="color:#42A5F5;text-decoration:none">zaravitamds18@gmail.com</a> ·
        +212 770 636 297
    </div>
</div>
""", unsafe_allow_html=True)