import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent/"models"

import joblib

def load_models():
    ridge = joblib.load(MODEL_DIR / "best_baseline_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    ridge_alpha = joblib.load(MODEL_DIR / "best_alpha_ridge.pkl")

    xgb = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    xgb_feature_cols = joblib.load(MODEL_DIR / "xgboost_feature_columns.pkl")

    shap_explainer = joblib.load(MODEL_DIR / "xgboost_shap_explainer.pkl")
    shap_values = joblib.load(MODEL_DIR / "xgboost_shap_values.pkl")

    print("✓ best_baseline_model.pkl loaded")
    print("✓ scaler.pkl loaded")
    print("✓ best_alpha_ridge.pkl loaded")
    print("✓ xgboost_model.pkl loaded")
    print("✓ feature_columns.pkl loaded")
    print("✓ xgboost_feature_columns.pkl loaded")
    print("✓ xgboost_shap_explainer.pkl loaded")
    print("✓ xgboost_shap_values.pkl loaded")

    return {
        "ridge": ridge,
        "scaler": scaler,
        "ridge_alpha": ridge_alpha,
        "xgb": xgb,
        "feature_cols": feature_cols,
        "xgb_feature_cols": xgb_feature_cols,
        "shap_explainer": shap_explainer,
        "shap_values": shap_values,
    }

def build_feature_row(data: dict, feature_cols: list) -> pd.DataFrame:
    d = data.copy()
 
    print("Input data received:", d)  # DEBUG

    d["TotalSF"] = (d.get("TotalBsmtSF", 0)
                    + d.get("1stFlrSF", 0)
                    + d.get("2ndFlrSF", 0)
                    )    
    d["HouseAge"]     = 2010 - d.get("YearBuilt", 2000)
    d["WasRemodeled"] = int(d.get("YearRemodAdd", 0) != d.get("YearBuilt", 0))
    d["RemodAge"]     = 2010 - d.get("YearRemodAdd", 2000)
    d["TotalBath"]    = d.get("FullBath", 0) + d.get("HalfBath", 0) * 0.5
    d["HasGarage"]    = int(d.get("GarageArea", 0) > 0)
    d["HasBasement"]  = int(d.get("TotalBsmtSF", 0) > 0)
    d["HasFireplace"] = int(d.get("Fireplaces", 0) > 0)
    d["HasPool"]      = 0
 
    neighborhood     = d.pop("Neighborhood", None)
    neighborhood_col = f"Neighborhood_{neighborhood}" if neighborhood else None
 
    row = pd.DataFrame([d])
    print("Row before reindex:", row.shape, row.dtypes.value_counts().to_dict())  # DEBUG

    if neighborhood_col and neighborhood_col in feature_cols:
        row[neighborhood_col] = 1
 
    row = row.reindex(columns=feature_cols, fill_value=0)
    print("Row after reindex:", row.shape)  # DEBUG
    print("Row sum:", row.sum().sum())  # DEBUG
    
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0)
 
    return row
 
 
def predict_price(data: dict, ridge, scaler, xgb, feature_cols: list) -> dict:
    MODEL_RMSE = 0.0694
 
    row        = build_feature_row(data, feature_cols)

    row_scaled = scaler.transform(row)
 
    ridge_pred = ridge.predict(row_scaled)[0]
    xgb_pred   = xgb.predict(row)[0]
 
    blended_log = (ridge_pred + xgb_pred) / 2
 
    price      = float(np.expm1(blended_log))
    price_low  = float(np.expm1(blended_log - MODEL_RMSE))
    price_high = float(np.expm1(blended_log + MODEL_RMSE))
 
    return {
        "predicted_price":      round(price, 2),
        "predicted_price_low":  round(price_low, 2),
        "predicted_price_high": round(price_high, 2),
        "model_rmse":           MODEL_RMSE
    }