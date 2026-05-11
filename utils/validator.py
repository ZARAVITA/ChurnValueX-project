"""
utils/validator.py — Dataset compatibility checker for ChurnIQ.
Returns a structured result: ok=True or ok=False + reasons list.
Never raises — always returns safely.
"""

import pandas as pd
import numpy as np

# Columns the pipeline actively uses (not all required, but checked for usability)
REQUIRED_COLS = {"CustomerID", "Churn", "Tenure"}

EXPECTED_COLS = {
    "CustomerID", "Churn", "Tenure", "CityTier", "WarehouseToHome",
    "HourSpendOnApp", "NumberOfDeviceRegistered", "SatisfactionScore",
    "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
    "Gender", "PreferredLoginDevice", "PreferredPaymentMode",
    "PreferedOrderCat", "MaritalStatus",
}


def validate_dataset(df) -> dict:
    """
    Returns dict:
      {
        "ok": bool,
        "score": int,          # 0-100 compatibility score
        "missing_required": list[str],
        "missing_optional": list[str],
        "present": list[str],
        "issues": list[str],   # human-readable problem descriptions
        "warnings": list[str], # non-blocking notices
      }
    """
    issues   = []
    warnings = []

    # ── 1. Empty dataframe ────────────────────────────────────────────────────
    if df is None or not isinstance(df, pd.DataFrame):
        return _fail(["Le dataset est vide ou illisible."])

    if len(df) == 0:
        return _fail(["Le dataset ne contient aucune ligne."])

    if len(df.columns) < 2:
        return _fail(["Le dataset ne contient pas assez de colonnes (minimum 2)."])

    # ── 2. Column presence ────────────────────────────────────────────────────
    cols = set(df.columns.tolist())
    missing_required = [c for c in sorted(REQUIRED_COLS) if c not in cols]
    missing_optional = [c for c in sorted(EXPECTED_COLS - REQUIRED_COLS) if c not in cols]
    present          = [c for c in sorted(EXPECTED_COLS) if c in cols]

    if missing_required:
        issues.append(
            f"Colonnes obligatoires manquantes : {', '.join(missing_required)}"
        )

    if len(missing_optional) > 10:
        issues.append(
            f"{len(missing_optional)} colonnes attendues sont absentes. "
            "La structure du dataset ne correspond pas au format ChurnIQ."
        )
    elif missing_optional:
        warnings.append(
            f"Colonnes optionnelles absentes ({len(missing_optional)}) : "
            f"{', '.join(missing_optional[:6])}"
            + (" …" if len(missing_optional) > 6 else "")
        )

    # ── 3. Churn column validity ──────────────────────────────────────────────
    if "Churn" in cols:
        try:
            churn_vals = df["Churn"].dropna().unique()
            churn_numeric = pd.to_numeric(pd.Series(churn_vals), errors="coerce")
            valid_churn = set(churn_numeric.dropna().astype(int).tolist())
            if not valid_churn.issubset({0, 1}):
                issues.append(
                    "La colonne 'Churn' doit contenir uniquement des valeurs 0 ou 1."
                )
        except Exception:
            issues.append("La colonne 'Churn' contient des valeurs non exploitables.")

    # ── 4. CustomerID validity ────────────────────────────────────────────────
    if "CustomerID" in cols:
        try:
            # Must be castable to int for pipeline key
            first_val = df["CustomerID"].dropna().iloc[0]
            int(first_val)
        except Exception:
            issues.append(
                "La colonne 'CustomerID' doit contenir des identifiants numériques entiers."
            )

    # ── 5. Tenure validity ───────────────────────────────────────────────────
    if "Tenure" in cols:
        try:
            tenure_num = pd.to_numeric(df["Tenure"], errors="coerce")
            if tenure_num.isna().all():
                issues.append("La colonne 'Tenure' ne contient aucune valeur numérique valide.")
        except Exception:
            issues.append("La colonne 'Tenure' contient des valeurs non exploitables.")

    # ── 6. Dataset size warning ───────────────────────────────────────────────
    if len(df) < 50:
        warnings.append(
            f"Dataset très petit ({len(df)} lignes). "
            "Les modèles ML requièrent au moins 50 observations pour des résultats fiables."
        )

    # ── 7. All-null columns ───────────────────────────────────────────────────
    null_cols = [c for c in REQUIRED_COLS & cols if df[c].isna().all()]
    if null_cols:
        issues.append(
            f"Colonnes entièrement vides : {', '.join(null_cols)}"
        )

    # ── Score ─────────────────────────────────────────────────────────────────
    score = int(100 * len(present) / max(len(EXPECTED_COLS), 1))

    ok = len(issues) == 0

    return {
        "ok":               ok,
        "score":            score,
        "present":          present,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "issues":           issues,
        "warnings":         warnings,
    }


def _fail(issues):
    return {
        "ok": False, "score": 0,
        "present": [], "missing_required": [], "missing_optional": [],
        "issues": issues, "warnings": [],
    }
