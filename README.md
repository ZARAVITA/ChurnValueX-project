# ⬡ ChurnIQ — Customer Churn Intelligence System

> **Système intelligent de pilotage de la rétention client**  
> Développé par **ZARA VITA** — Smart Automation Technologies

---

## Aperçu

ChurnIQ est une application analytique de niveau enterprise qui transforme des données clients brutes en stratégie de rétention actionnable. Le système combine quatre modèles complémentaires pour répondre à une question centrale : **quels clients allons-nous perdre, quelle valeur représentent-ils, et que faire maintenant ?**

L'interface est conçue pour des équipes Marketing, CRM et direction — non pour des data scientists. Chaque résultat est traduit en recommandation business claire.

---

## Fonctionnalités

| Onglet | Contenu |
|--------|---------|
| 🏠 Vue d'ensemble | KPIs clés, distributions CLV/risque, comparaison avant/après système |
| 📊 Indicateurs | Analyse approfondie : facteurs de churn, courbes de survie, statistiques par segment |
| 🎯 Matrice de décision | Quadrant CLV × Risque, 4 segments d'action avec recommandations |
| 👥 Clients | Liste interactive filtrée, triable, avec export Excel par segment |
| 💡 Insights & Simulation | Insights consulting auto-générés + simulateur what-if ROI |

**Autres fonctionnalités :**
- 🌙 Mode sombre / ☀️ Mode clair — toggle en temps réel
- 📂 Dataset de démonstration intégré (ou données synthétiques si fichier absent)
- ⬇️ Export Excel : clients PRIORITY, à risque 30 jours, top CLV
- ℹ️ Documentation des données dans la sidebar

---

## Architecture du projet

```
ChurnValueX-project/
│
├── app.py                        # Point d'entrée — streamlit run app.py
│
├── config/
│   ├── __init__.py
│   └── theme.py                  # Tokens design light & dark (couleurs, fonds, accents)
│
├── models/
│   ├── __init__.py
│   └── pipeline.py               # Pipeline ML complet (mis en cache avec @st.cache_data)
│                                 #   _clean → _segment → _predict_churn
│                                 #   → _predict_survival → _predict_clv → _assign_segment
│
├── ui/
│   ├── __init__.py
│   ├── styles.py                 # Générateur CSS dynamique (light/dark) — tous les composants
│   ├── sidebar.py                # Sidebar : toggle thème, chargement data, documentation colonnes
│   ├── charts.py                 # Tous les graphiques Plotly (thème-aware)
│   └── tabs/
│       ├── __init__.py
│       ├── tab_overview.py       # Onglet 1 : Hero + KPIs + avant/après
│       ├── tab_analytics.py      # Onglet 2 : Indicateurs & facteurs de churn
│       ├── tab_matrix.py         # Onglet 3 : Matrice de décision 4 quadrants
│       ├── tab_customers.py      # Onglet 4 : Liste clients filtrée + exports
│       └── tab_insights.py       # Onglet 5 : Insights consulting + simulation what-if
│
├── utils/
│   ├── __init__.py
│   ├── export.py                 # Helper export Excel (BytesIO + openpyxl)
│   └── synthetic.py              # Générateur de données synthétiques réalistes (900 clients)
│
├── data/
│   └── E Commerce Dataset.xlsx   # Dataset de démonstration (non inclus dans le repo)
│
├── requirements.txt
└── README.md
```

---

## Modèles analytiques

### 1. Segmentation — K-Means Clustering
Segmentation non supervisée des clients en 2 clusters sur l'ensemble des variables comportementales (StandardScaler + KMeans). Utilisé comme covariate dans les modèles suivants.

### 2. Prédiction du churn — XGBoost Classifier
- Équilibrage des classes par **SMOTE** (sur-échantillonnage de la classe minoritaire)
- Split 80/20 stratifié
- 70 estimateurs, `eval_metric=logloss`
- Sortie : probabilité de churn individuelle `Churn_proba ∈ [0, 1]`

### 3. Analyse de survie — Weibull AFT
- **Accelerated Failure Time** avec distribution de Weibull (via `lifelines`)
- Variable duration : `Tenure` (ancienneté en mois)
- Variable événement : `Churn`
- Sorties : probabilité de churn dans les **30 jours** et **90 jours** pour chaque client

### 4. Valeur vie client — BG/NBD + Gamma-Gamma
- **Beta-Geometric/Negative Binomial** (via `lifetimes`) : modélise la fréquence d'achat future
- **Gamma-Gamma** : modélise la valeur monétaire moyenne par transaction
- CLV projetée sur **12 mois** avec taux d'actualisation de 1%
- Les transactions sont reconstruites à partir des données agrégées (Tenure, OrderCount, CashbackAmount)

### 5. Matrice de décision
Croisement CLV (médiane comme seuil) × Risque de churn (seuil 50%) → 4 segments :

| Segment | CLV | Risque | Action |
|---------|-----|--------|--------|
| 🔴 **PRIORITY** | Élevée | Élevé | Contact personnel immédiat + offre exclusive |
| 🟢 **PROTECT** | Élevée | Faible | Programme VIP — consolider la relation |
| 🟡 **OPTIMIZE** | Faible | Élevé | Campagne email/coupon — vérifier le ROI |
| ⚫ **AUTOMATE** | Faible | Faible | Réengagement automatisé à faible coût |

---

## Installation

### Prérequis
- Python 3.9 — 3.11
- pip

### Installation rapide

```bash
# 1. Cloner le repo
git clone https://github.com/zaravita/ChurnValueX-project.git
cd ChurnValueX-project

# 2. Créer un environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

---

## Dépendances

```
streamlit==1.32.2
pandas==2.2.2
numpy==1.26.4
plotly==5.22.0
scikit-learn==1.3.2
imbalanced-learn==0.11.0
xgboost==2.0.3
scipy==1.10.1
lifelines==0.27.8
lifetimes==0.11.3
matplotlib==3.8.4
openpyxl==3.1.2
```

---

## Format des données d'entrée

Le système est conçu pour des données clients e-commerce / CRM structurées. Le dataset doit contenir les colonnes suivantes (ou un sous-ensemble adaptable) :

| Colonne | Type | Description |
|---------|------|-------------|
| `CustomerID` | int | Identifiant unique client |
| `Churn` | int | Cible : 1 = churné, 0 = retenu |
| `Tenure` | float | Ancienneté en mois |
| `CityTier` | int | Niveau de ville : 1 (métropole) → 3 (rural) |
| `WarehouseToHome` | int | Distance entrepôt → domicile (km) |
| `HourSpendOnApp` | float | Heures/semaine sur l'app |
| `NumberOfDeviceRegistered` | int | Appareils enregistrés |
| `SatisfactionScore` | int | Score satisfaction (1–5) |
| `NumberOfAddress` | int | Adresses de livraison sauvegardées |
| `Complain` | int | 1 = réclamation déposée ce mois |
| `OrderAmountHikeFromlastYear` | float | Hausse % du montant commandé vs an dernier |
| `CouponUsed` | int | Coupons utilisés ce mois |
| `OrderCount` | int | Commandes passées ce mois |
| `DaySinceLastOrder` | int | Jours depuis la dernière commande |
| `CashbackAmount` | float | Cashback reçu ce mois (€/$) |
| `Gender` | str | Male / Female |
| `PreferredLoginDevice` | str | Mobile Phone / Computer / Phone |
| `PreferredPaymentMode` | str | Debit Card / Credit Card / UPI / … |
| `PreferedOrderCat` | str | Laptop & Accessory / Mobile / Fashion / … |
| `MaritalStatus` | str | Married / Single / Divorced |

> **Le système est entièrement adaptable** : les colonnes, seuils et paramètres des modèles peuvent être reconfigurés dans `models/pipeline.py` pour s'aligner sur les données spécifiques de chaque entreprise.

---

## Utilisation sans dataset

Si le fichier `data/E Commerce Dataset.xlsx` est absent, le système génère automatiquement un dataset synthétique réaliste de 900 clients (`utils/synthetic.py`) pour permettre une démonstration immédiate.

---

## Contact

**ZARA VITA**  
Smart Automation Technologies  
📧 zaravitamds18@gmail.com  
📱 +212 770 636 297

---

*ChurnIQ — Transformez vos données clients en stratégie de rétention.*