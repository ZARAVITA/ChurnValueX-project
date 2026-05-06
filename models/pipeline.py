"""
models/pipeline.py — Full analytics pipeline:
  _clean → _segment → _predict_churn → _predict_survival → _predict_clv → _assign_segment
"""

import numpy as np
import pandas as pd
import streamlit as st

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


# ── Public entry point ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _clean(df_raw.copy())
    df = _segment(df)
    df, _model, _features = _predict_churn(df)
    df = _predict_survival(df)
    df, clv_df = _predict_clv(df_raw.copy(), df)
    df = df.merge(clv_df, on="CustomerID", how="left")
    df["CLV_12mois"] = df["CLV_12mois"].fillna(0).clip(lower=0)
    df["SPA_score"] = df["Churn_proba"] * df["CLV_12mois"]
    df = _assign_segment(df)
    return df


# ── Step 1: Clean ─────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Tenure", "OrderCount", "CashbackAmount", "HourSpendOnApp", "DaySinceLastOrder"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Tenure" in df.columns:
        df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median()).replace(0, 0.001)
    if "DaySinceLastOrder" in df.columns:
        df["DaySinceLastOrder"] = df["DaySinceLastOrder"].fillna(df["DaySinceLastOrder"].max())
    if "HourSpendOnApp" in df.columns:
        df["HourSpendOnApp"] = df["HourSpendOnApp"].fillna(df["HourSpendOnApp"].median())
    if "OrderCount" in df.columns:
        df["OrderCount"] = df["OrderCount"].fillna(0).clip(lower=0)
    if "CashbackAmount" in df.columns:
        df["CashbackAmount"] = df["CashbackAmount"].fillna(df["CashbackAmount"].median())

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"].astype(str))

    cats = [c for c in ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat", "MaritalStatus"] if c in df.columns]
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=True)

    df[df.select_dtypes(include=["bool"]).columns] = df.select_dtypes(include=["bool"]).astype(int)
    return df


# ── Step 2: Segment ───────────────────────────────────────────────────────────

def _segment(df: pd.DataFrame, n_clusters: int = 2) -> pd.DataFrame:
    X = df.drop(columns=["Churn", "CustomerID"], errors="ignore")
    sc = StandardScaler()
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    df["segment"] = km.fit_predict(sc.fit_transform(X))
    return df


# ── Step 3: Churn (XGBoost) ───────────────────────────────────────────────────

def _predict_churn(df: pd.DataFrame):
    features = [c for c in df.columns if c not in ["CustomerID", "Churn"]]
    X = df[features]
    y = df.get("Churn", pd.Series(np.zeros(len(df)), index=df.index))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2,
        stratify=y if y.nunique() > 1 else None,
        random_state=SEED
    )
    if y.nunique() > 1:
        sm = SMOTE(random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                               random_state=SEED, n_estimators=70)
    model.fit(X_tr, y_tr)
    df["Churn_proba"] = model.predict_proba(df[features])[:, 1]
    df["Churn_pred"]  = model.predict(df[features])
    return df, model, features


# ── Step 4: Survival (Weibull AFT) ────────────────────────────────────────────

def _predict_survival(df: pd.DataFrame) -> pd.DataFrame:
    dur, evt = "Tenure", "Churn"
    df[dur] = df[dur].replace(0, 0.001)
    covs = [c for c in [
        "CityTier", "WarehouseToHome", "Gender", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount",
        "PreferredLoginDevice_Mobile Phone", "PreferredLoginDevice_Phone",
        "PreferredPaymentMode_COD", "PreferredPaymentMode_Cash on Delivery",
        "PreferredPaymentMode_Credit Card", "PreferredPaymentMode_Debit Card",
        "PreferredPaymentMode_E wallet", "PreferredPaymentMode_UPI",
        "PreferedOrderCat_Grocery", "PreferedOrderCat_Laptop & Accessory",
        "PreferedOrderCat_Mobile", "PreferedOrderCat_Mobile Phone",
        "PreferedOrderCat_Others", "MaritalStatus_Single", "segment"
    ] if c in df.columns]

    aft = WeibullAFTFitter()
    aft.fit(df[[dur, evt] + covs].dropna(), duration_col=dur, event_col=evt)

    s0 = aft.predict_survival_function(df[covs], times=df[dur]).T
    s1 = aft.predict_survival_function(df[covs], times=df[dur] + 1).T
    s3 = aft.predict_survival_function(df[covs], times=df[dur] + 3).T

    df["proba_churn_1mois"] = (s0.values[:, 0] - s1.values[:, 0]).clip(0, 1)
    df["proba_churn_3mois"] = (s0.values[:, 0] - s3.values[:, 0]).clip(0, 1)
    df["proba_churn_now"]   = (1 - s0.values[:, 0]).clip(0, 1)
    return df


# ── Step 5: CLV (BG/NBD + Gamma-Gamma) ───────────────────────────────────────

def _predict_clv(df_raw: pd.DataFrame, df_proc: pd.DataFrame):
    df = df_raw.copy()
    for col in ["Tenure", "OrderCount", "CashbackAmount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median()).replace(0, 0.001)
    if "OrderCount" in df.columns:
        df["OrderCount"] = df["OrderCount"].fillna(0).clip(lower=0)
    if "CashbackAmount" in df.columns:
        df["CashbackAmount"] = df["CashbackAmount"].fillna(df["CashbackAmount"].median())

    df["Tenure_days"]   = (df["Tenure"] * 30).round().fillna(0).astype(int)
    df_c = df[df["Tenure_days"] >= 50].copy()
    df_c["Tenure_months"] = (df_c["Tenure_days"] / 30).round().fillna(1).astype(int)

    analysis_date = pd.Timestamp("2025-11-18")
    txns = []
    for _, row in df_c.iterrows():
        tenure_m = int(row["Tenure_months"]) if pd.notna(row["Tenure_months"]) else 1
        order_c  = float(row["OrderCount"])  if ("OrderCount" in row and pd.notna(row["OrderCount"])) else 1.0
        tenure_d = int(row["Tenure_days"])   if pd.notna(row["Tenure_days"]) else 30
        n = max(1, int(tenure_m * max(order_c, 1)))

        start  = analysis_date - pd.to_timedelta(tenure_d, unit="D")
        dates  = pd.date_range(start=start, end=analysis_date, periods=n)
        dates  = dates + pd.to_timedelta(np.random.randint(-7, 7, n), unit="D")

        cashback = float(row["CashbackAmount"]) if ("CashbackAmount" in row and pd.notna(row["CashbackAmount"])) else 5.0
        mean_amt = (cashback / 0.05) / max(1.0, order_c)
        amounts  = np.random.gamma(2, max(mean_amt / 2, 0.01), n)

        for d, a in zip(dates, amounts):
            txns.append([row["CustomerID"], d, max(0.01, a)])

    if not txns:
        return df_proc, pd.DataFrame(columns=["CustomerID", "CLV_12mois"])

    df_txn = pd.DataFrame(txns, columns=["CustomerID", "Date", "Amount"])
    summary = summary_data_from_transaction_data(
        df_txn, "CustomerID", "Date", "Amount", observation_period_end=analysis_date
    )
    summary = summary[summary["monetary_value"] > 0]
    sumg    = summary[summary["frequency"] > 0].copy()

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(sumg["frequency"], sumg["monetary_value"])

    clv = ggf.customer_lifetime_value(
        bgf, summary["frequency"], summary["recency"],
        summary["T"], summary["monetary_value"],
        time=12, freq="D", discount_rate=0.01
    ).reset_index()
    clv.columns = ["CustomerID", "CLV_12mois"]
    clv["CLV_12mois"] = (clv["CLV_12mois"] / 10).clip(lower=0)
    return df_proc, clv


# ── Step 6: Decision matrix segment ──────────────────────────────────────────

def _assign_segment(df: pd.DataFrame) -> pd.DataFrame:
    clv_med      = df["CLV_12mois"].median()
    risk_thresh  = 0.5
    df["Matrix_Segment"] = np.select(
        [
            (df["Churn_proba"] >= risk_thresh) & (df["CLV_12mois"] >= clv_med),
            (df["Churn_proba"] <  risk_thresh) & (df["CLV_12mois"] >= clv_med),
            (df["Churn_proba"] >= risk_thresh) & (df["CLV_12mois"] <  clv_med),
            (df["Churn_proba"] <  risk_thresh) & (df["CLV_12mois"] <  clv_med),
        ],
        ["PRIORITY", "PROTECT", "OPTIMIZE", "AUTOMATE"],
        default="AUTOMATE"
    )
    return df
