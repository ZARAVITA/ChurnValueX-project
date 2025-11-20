# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:09:36 2025

@author: ZARA VITA
"""


# app.py
# Streamlit app pour Pilotage de la Rétention Client
# Usage: streamlit run app.py
import os
os.environ["STREAMLIT_USE_ARROW"] = "0"   # désactive PyArrow
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# Adaptation des fonctions de nettoyage, segmentation, modélisation, AFT et CLV
# depuis fichier MiseEnProductionPFE.py (intégrées ci-dessous).
# Référence : MiseEnProductionPFE.py. :contentReference[oaicite:1]{index=1}

# ---------- Utilities (clean_data, segmentation, predict_churn, predict_churn_proba_x_mois, predict_clv) ----------
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from lifelines import WeibullAFTFitter
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import matplotlib.pyplot as plt



SEED = 42
np.random.seed(SEED)

# --- clean_data ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # filling common cols as in original script
    if 'Tenure' in df.columns:
        df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
        df['Tenure'] = df['Tenure'].replace(0, 0.001)
    if 'DaySinceLastOrder' in df.columns:
        df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].max())
    if 'HourSpendOnApp' in df.columns:
        df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())
    df['OrderCount'] = df.get('OrderCount', pd.Series()).fillna(0).clip(lower=0) if 'OrderCount' in df.columns else df.get('OrderCount', 0)
    if 'CashbackAmount' in df.columns:
        df['CashbackAmount'] = df['CashbackAmount'].fillna(df['CashbackAmount'].median())
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    # encoding
    if 'Gender' in df.columns:
        le = LabelEncoder()
        try:
            df['Gender'] = le.fit_transform(df['Gender'].astype(str))
        except:
            df['Gender'] = df['Gender'].astype(int)
    cat_vars = [c for c in ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus'] if c in df.columns]
    if cat_vars:
        df = pd.get_dummies(df, columns=cat_vars, drop_first=True)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

# --- segmentation_kmeans ---
def segmentation_kmeans(df: pd.DataFrame, n_clusters:int=2):
    df = df.copy()
    X = df.drop(columns=['Churn', 'CustomerID'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = kmeans.fit_predict(X_scaled)
    df['segment'] = labels
    return df

# --- predict_churn (XGBoost) ---
def predict_churn(df: pd.DataFrame, smote=True):
    df = df.copy()
    features = [c for c in df.columns if c not in ['CustomerID','Churn']]
    # remove obviously problematic columns if present
    for bad in ["PreferedOrderCat_Others", "PreferedOrderCat_Grocery"," PreferredPaymentMode_Cash on Delivery"]:
        if bad in features:
            features.remove(bad)
    X = df[features]
    y = df['Churn'] if 'Churn' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y if y.nunique()>1 else None, random_state=SEED)
    if smote and y.nunique() > 1:
        sm = SMOTE(random_state=SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED, n_estimators=70)  #=====================
    model.fit(X_train, y_train)
    # probabilities for all rows
    df['Churn_proba'] = model.predict_proba(df[features])[:,1]
    df['Churn_pred'] = model.predict(df[features])
    df['retention_rate'] = 1 - df['Churn_proba']
    return df, model, features

# --- predict_churn_proba_x_mois (Weibull AFT as in original) ---
def predict_churn_proba_x_mois(df: pd.DataFrame):
    df = df.copy()
    duration_col = 'Tenure'
    event_col = 'Churn'
    df[duration_col] = df[duration_col].replace(0, 0.001)
    # define covariates present in df
    covariates = [c for c in [
        'CityTier','WarehouseToHome','Gender','HourSpendOnApp','NumberOfDeviceRegistered',
        'SatisfactionScore','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed',
        'OrderCount','DaySinceLastOrder','CashbackAmount','PreferredLoginDevice_Mobile Phone',
        'PreferredLoginDevice_Phone','PreferredPaymentMode_COD','PreferredPaymentMode_Cash on Delivery',
        'PreferredPaymentMode_Credit Card','PreferredPaymentMode_Debit Card','PreferredPaymentMode_E wallet',
        'PreferredPaymentMode_UPI','PreferedOrderCat_Grocery','PreferedOrderCat_Laptop & Accessory',
        'PreferedOrderCat_Mobile','PreferedOrderCat_Mobile Phone','PreferedOrderCat_Others','MaritalStatus_Single','segment'
    ] if c in df.columns]
    # build df for aft
    df_aft = df[[duration_col, event_col] + covariates].dropna()
    aft = WeibullAFTFitter()
    aft.fit(df_aft, duration_col=duration_col, event_col=event_col)
    # horizons
    horizon_1 = df[duration_col] + 1
    horizon_3 = df[duration_col] + 3
    surv_now = aft.predict_survival_function(df[covariates], times=df[duration_col]).T
    surv_1 = aft.predict_survival_function(df[covariates], times=horizon_1).T
    surv_3 = aft.predict_survival_function(df[covariates], times=horizon_3).T
    df["proba_churn_1mois"] = (surv_now.values[:, 0] - surv_1.values[:, 0]).clip(0,1)
    df["proba_churn_3mois"] = (surv_now.values[:, 0] - surv_3.values[:, 0]).clip(0,1)
    # immediate churn probability
    t_now = df[duration_col]
    surv_now2 = aft.predict_survival_function(df[covariates], times=t_now).T
    df["proba_churn_now"] = 1 - surv_now2.values[:,0]
    df["churn_pred_now"] = (df["proba_churn_now"] >= 0.5).astype(int)
    return df, aft

# --- predict_clv (BG/NBD + GammaGamma) ---
def predict_clv(df: pd.DataFrame, analysis_date_str="2025-11-18"):
    df = df.copy()
    # original code multiplies Tenure by 30 to get days
    if 'Tenure' in df.columns:
        df['Tenure_days'] = (df['Tenure'] * 30).astype(int)
    else:
        df['Tenure_days'] = 30
    # prefered order cat column if exists
    if 'PreferedOrderCat' in df.columns:
        df['PreferedOrderCat'] = df['PreferedOrderCat'].astype(str)
    # filter as in original
    df_clean = df[(df['Tenure_days'] >= 50)].copy()
    df_clean['Tenure_months'] = (df_clean['Tenure_days'] / 30).round().astype(int)
    # estimated transactions heuristic (preserved)
    mean_order_by_tenure = df_clean.groupby('Tenure_months')['OrderCount'].mean().to_dict() if 'OrderCount' in df_clean.columns else {}
    def estimate_transactions(row):
        tenure_months = row['Tenure_months']
        order_count = row.get('OrderCount',0)
        est_basic = tenure_months * order_count
        est_max = max(order_count, est_basic)
        if order_count == 0 and row['Tenure_days'] >= 50:
            mean_orders = mean_order_by_tenure.get(tenure_months, df_clean['OrderCount'].mean() if 'OrderCount' in df_clean.columns else 1)
            return int(tenure_months * mean_orders)
        return int(est_max)
    df_clean['EstimatedTransactions'] = df_clean.apply(estimate_transactions, axis=1)
    # keep at least 1
    df_clean['EstimatedTransactions'] = df_clean['EstimatedTransactions'].apply(lambda x: max(int(x), 1))
    transactions = []
    analysis_date = pd.to_datetime(analysis_date_str)
    for _, row in df_clean.iterrows():
        n = int(row['EstimatedTransactions'])
        if n < 1:
            continue
        end_date = analysis_date
        start_date = end_date - pd.to_timedelta(int(row['Tenure_days']), unit='D')
        dates = pd.date_range(start=start_date, end=end_date, periods=n)
        noise1 = pd.to_timedelta(np.random.randint(-10,10,size=n), unit='D')
        dates = dates + noise1
        # estimate monetary amounts
        cashback_rates = {
            'Laptop & Accessory': 0.05, 'Mobile': 0.05, 'Mobile Phone': 0.05, 'Fashion': 0.05, 'Grocery': 0.051, 'Others': 0.05
        }
        cat = row.get('PreferedOrderCat','Others')
        cashback_rate = cashback_rates.get(cat, 0.05)
        mean_amount = (row.get('CashbackAmount', 1) / cashback_rate) / max(1, row.get('OrderCount',1))
        amounts = np.random.gamma(shape=2.0, scale=max(mean_amount/2.0, 0.01), size=n)
        for d, amt in zip(dates, amounts):
            transactions.append([row['CustomerID'], d, max(0.01, amt)])
    df_txn = pd.DataFrame(transactions, columns=["CustomerID","Date","Amount"])
    if df_txn.empty:
        # return empty clv
        return df_clean, pd.DataFrame(columns=["CustomerID","CLV"]), pd.DataFrame(), []
    df_txn['Date'] = pd.to_datetime(df_txn['Date'])
    summary = summary_data_from_transaction_data(df_txn, customer_id_col="CustomerID", datetime_col="Date", monetary_value_col="Amount", observation_period_end=analysis_date)
    summary = summary[summary['monetary_value'] > 0]
    summary_gg = summary[summary['frequency'] > 0].copy()
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(summary_gg['frequency'], summary_gg['monetary_value'])
    clv = ggf.customer_lifetime_value(bgf, summary['frequency'], summary['recency'], summary['T'], summary['monetary_value'], time=12, freq='D', discount_rate=0.01)
    clv = clv.reset_index()
    clv.columns = ["CustomerID","CLV"]
    clv = clv[clv>0]
    return df_clean, clv, summary, transactions

# ---------- Helper pour download excel ----------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()

# ---------- UI ----------

st.set_page_config(page_title="Churn & CLV Dashboard", layout="wide")
st.title("Système intelligent de pilotage de la rétention client")
st.markdown(
    """
    Ce système développé au sein de **Smart Automation Technologies** par **ZARA VITA** offre une plateforme intelligente de pilotage de la **rétention client**.  
    Il réunit **segmentation automatique**, **prédiction du churn via XGBoost**, **analyse de survie (Weibull AFT)** pour anticiper les risques à 1–3 mois,  
    et **estimation avancée de la CLV (BG/NBD + Gamma-Gamma)**. L’interface fournit des **KPI clés**, une **vision temps réel des clients prioritaires** et des **recommandations actionnables** pour maximiser la rétention, la valeur client et la rentabilité globale.

    Le système prend en entrée des **données clients structurées et spécifiques** (profil, historique de commandes, interactions, paiements, etc.) et applique en interne ses **méthodes analytiques avancées** pour générer des résultats opérationnels. Une **démonstration intégrée** permet de lancer la démo et tester immédiatement le pipeline : les jeux de données fournis sont optimisés pour ce système, mais restent **entièrement personnalisables** aux données réelles de chaque entreprise.
    """
)

# Sidebar: data source
st.sidebar.header("Source des données")
data_choice = st.sidebar.radio("Choisir une option :", ("Lancer la démo avec le dataset intégré", "Importer un fichier CSV/Excel"))

uploaded_df = None
sample_path = "data/E Commerce Dataset.xlsx"
if data_choice == "Lancer la démo avec le dataset intégré":
    st.sidebar.markdown("Utilisez le dataset d'exemple stocké dans le système.")
    use_sample = st.sidebar.button("Charger le dataset d'exemple")
    if use_sample:
        try:
            uploaded_df = pd.read_excel(sample_path, sheet_name=1)
            st.success("Dataset d'exemple chargé.")
        except Exception as e:
            st.error(f"Impossible de charger le dataset exemple : {e}")
else:
    file = st.sidebar.file_uploader("Importer CSV ou Excel", type=["csv","xlsx"])
    if file is not None:
        try:
            if file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(file)
            else:
                uploaded_df = pd.read_excel(file, sheet_name=0)
            st.success("Fichier importé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")

if uploaded_df is None:
    st.info("Aucun dataset chargé. Chargez l'exemple ou importez votre fichier pour lancer l'analyse.")
    st.stop()

# Preprocess + Pipeline
with st.spinner("Nettoyage et préparation des données..."):
    df_clean = clean_data(uploaded_df)

with st.spinner("Segmentation et prédiction (XGBoost)..."):
    try:
        df_seg = segmentation_kmeans(df_clean)
        df_pred, xgb_model, features = predict_churn(df_seg)
    except Exception as e:
        st.error(f"Erreur lors de la segmentation/prediction : {e}")
        st.stop()

with st.spinner("Analyse de survie (Weibull AFT) et prédiction 1/3 mois..."):
    try:
        df_aft_res, aft_model = predict_churn_proba_x_mois(df_pred)
    except Exception as e:
        st.error(f"Erreur AFT : {e}")
        st.stop()

with st.spinner("Calcul CLV (BG/NBD + Gamma-Gamma)..."):
    try:
        df_clv_clean, clv_df, summary, transactions = predict_clv(df_clean)
    except Exception as e:
        st.error(f"Erreur CLV : {e}")
        clv_df = pd.DataFrame(columns=["CustomerID","CLV"])

# Merge CLV into final
df_final = df_aft_res.copy()
if not clv_df.empty:
    clv_df.rename(columns={"CLV":"CLV_12mois"}, inplace=True)
    df_final = df_final.merge(clv_df, on="CustomerID", how="left")
    df_final["CLV_12mois"] = df_final["CLV_12mois"].fillna(0) / 10.0
else:
    df_final["CLV_12mois"] = 0.0

# Priority score SPA = churn_proba * CLV
df_final["SPA_score"] = df_final["Churn_proba"] * df_final["CLV_12mois"]

# Tabs: Indicateurs / Résultats & Reco / Téléchargements
tab1, tab2, tab3 = st.tabs(["Indicateurs", "Résultats & Recommandations", "Téléchargements"])

# --- Tab 1: KPIs & Graphiques ---
with tab1:
    st.header("Indicateurs clés")
    c1, c2, c3, c4 = st.columns(4)
    total_clients = len(df_final)
    churn_global = (uploaded_df['Churn'].mean() if 'Churn' in uploaded_df.columns else df_final['Churn_pred'].mean())
    churn_pred_mean = df_final['Churn_proba'].mean()
    mean_clv = df_final['CLV_12mois'].mean()
    c1.metric("Total clients", f"{total_clients}")
    c2.metric("Taux churn (observé)", f"{churn_global:.2%}")
    c3.metric("Taux churn (prévu, moyen)", f"{churn_pred_mean:.2%}")
    c4.metric("CLV moyen (12 mois)", f"{mean_clv:.2f}")

    st.markdown("### Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_final, x="CLV_12mois", nbins=50, title="Histogramme du CLV (12 mois)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_final, x="Churn_proba", nbins=50, title="Distribution de la probabilité de churn")
        st.plotly_chart(fig2, use_container_width=True)

    # camembert des segments
    if 'segment' in df_final.columns:
        seg_counts = df_final['segment'].value_counts().reset_index()
        seg_counts.columns = ['segment','count']
        fig3 = px.pie(seg_counts, names='segment', values='count', title="Répartition par segments")
        st.plotly_chart(fig3, use_container_width=True)

# --- Tab 2: Résultats & Recommandations ---
with tab2:
    st.header("Cibles & Recommandations")
    st.markdown("**1) Clients à risque élevé (immédiat)**")
    high_risk_now = df_final.sort_values("proba_churn_now", ascending=False).head(200)
    st.metric("Nombre clients à risque élevé (proche)", f"{(df_final['proba_churn_now']>=0.5).sum()}")
    st.dataframe(high_risk_now[["CustomerID","proba_churn_now","Churn_proba","CLV_12mois","SPA_score"]].head(50))

    st.markdown("**2) Clients à risque dans 1 mois / 3 mois**")
    risk_1m = df_final.sort_values("proba_churn_1mois", ascending=False).head(200)
    risk_3m = df_final.sort_values("proba_churn_3mois", ascending=False).head(200)
    st.write("Clients à risque (1 mois) :", len(risk_1m))
    st.dataframe(risk_1m[["CustomerID","proba_churn_1mois","Churn_proba","CLV_12mois"]].head(30))
    st.write("Clients à risque (3 mois) :", len(risk_3m))
    st.dataframe(risk_3m[["CustomerID","proba_churn_3mois","Churn_proba","CLV_12mois"]].head(30))

    st.markdown("**3) Clients à forte CLV**")
    top_clv = df_final.sort_values("CLV_12mois", ascending=False).head(200)
    st.dataframe(top_clv[["CustomerID","CLV_12mois","Churn_proba"]].head(50))

    st.markdown("**4) Combinaison : Risque élevé × CLV élevé (Score de priorité SPA)**")
    top_spa = df_final.sort_values("SPA_score", ascending=False).head(200)
    st.dataframe(top_spa[["CustomerID","SPA_score","Churn_proba","CLV_12mois"]].head(50))

    st.markdown("### Recommandations automatiques (exemples générés)")
    def rec_for_row(row):
        if row['Churn_proba']>0.7 and row['CLV_12mois']>np.percentile(df_final['CLV_12mois'].replace(0,np.nan).dropna(), 75):
            return "Client très rentable mais à haut risque — action urgente : contact personnalisé + incentive."
        if row['Churn_proba']>0.5 and row['CLV_12mois']>0:
            return "Client à risque — campagne d'emailing + coupon ciblé."
        if row['CLV_12mois']>np.percentile(df_final['CLV_12mois'].replace(0,np.nan).dropna(), 90):
            return "Client à garder absolument — proposer programme de fidélité VIP."
        return "Surveiller / remarketing peu coûteux."
    sample_recs = top_spa.head(10).copy()
    sample_recs['Recommendation'] = sample_recs.apply(rec_for_row, axis=1)
    st.table(sample_recs[["CustomerID","SPA_score","Churn_proba","CLV_12mois","Recommendation"]])

# --- Tab 3: Téléchargements ---
with tab3:
    st.header("Téléchargements")
    st.markdown("Téléchargez les listes ciblées pour les actions marketing.")

    # immediate risk
    df_risk_now = df_final.sort_values("proba_churn_now", ascending=False)[["CustomerID","proba_churn_now","Churn_proba","CLV_12mois","SPA_score"]].head(500)
    df_risk_1m = df_final.sort_values("proba_churn_1mois", ascending=False)[["CustomerID","proba_churn_1mois","Churn_proba","CLV_12mois"]].head(500)
    df_risk_3m = df_final.sort_values("proba_churn_3mois", ascending=False)[["CustomerID","proba_churn_3mois","Churn_proba","CLV_12mois"]].head(500)
    df_top_clv = df_final.sort_values("CLV_12mois", ascending=False)[["CustomerID","CLV_12mois","Churn_proba"]].head(500)

    b1 = to_excel_bytes(df_risk_now)
    b2 = to_excel_bytes(pd.concat([df_risk_1m, df_risk_3m], keys=["1M","3M"]))
    b3 = to_excel_bytes(df_top_clv)

    st.download_button("Télécharger : Clients les plus à risque (immédiat)", data=b1, file_name="clients_risque_immediat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Télécharger : Clients risque 1 mois / 3 mois", data=b2, file_name="clients_risque_1_3_mois.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Télécharger : Clients plus forte valeur (CLV)", data=b3, file_name="clients_top_clv.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("**Notes techniques :** Modèles XGBoost (classification), Weibull AFT (survie), BG/NBD + Gamma-Gamma (CLV). Les paramètres et seuils peuvent être ajustés dans le code si nécessaire.")









