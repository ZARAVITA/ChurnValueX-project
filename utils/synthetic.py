"""
utils/synthetic.py — Realistic synthetic e-commerce dataset generator.
"""

import numpy as np
import pandas as pd

SEED = 42


def generate_synthetic_data(n: int = 900) -> pd.DataFrame:
    np.random.seed(SEED)
    return pd.DataFrame({
        "CustomerID":                   range(10000, 10000 + n),
        "Churn":                        np.random.choice([0, 1], n, p=[0.83, 0.17]),
        "Tenure":                       np.random.gamma(3, 5, n).clip(1, 60).round(1),
        "CityTier":                     np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
        "WarehouseToHome":              np.random.randint(5, 40, n),
        "HourSpendOnApp":               np.random.gamma(2, 1.5, n).clip(0, 8).round(1),
        "NumberOfDeviceRegistered":     np.random.randint(1, 6, n),
        "SatisfactionScore":            np.random.randint(1, 6, n),
        "NumberOfAddress":              np.random.randint(1, 10, n),
        "Complain":                     np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "OrderAmountHikeFromlastYear":  np.random.normal(15, 8, n).clip(0, 50).round(1),
        "CouponUsed":                   np.random.randint(0, 15, n),
        "OrderCount":                   np.random.randint(1, 20, n),
        "DaySinceLastOrder":            np.random.randint(0, 30, n),
        "CashbackAmount":               np.random.gamma(4, 50, n).round(2),
        "Gender":                       np.random.choice(["Male", "Female"], n),
        "PreferredLoginDevice":         np.random.choice(["Mobile Phone", "Computer", "Phone"], n, p=[0.6, 0.2, 0.2]),
        "PreferredPaymentMode":         np.random.choice(["Debit Card", "Credit Card", "UPI", "E wallet", "Cash on Delivery"], n, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        "PreferedOrderCat":             np.random.choice(["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"], n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        "MaritalStatus":                np.random.choice(["Married", "Single", "Divorced"], n, p=[0.55, 0.33, 0.12]),
    })
