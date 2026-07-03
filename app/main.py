from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn

#app setup
app = FastAPI(
    title="House Price Prediction API",
    description="Predicts Ames Iowa house prices using a Ridge + XGBoost blend model",
    version="1.0.0"
)

#frontend tp talk to api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#model loading
MODEL_DIR = Path(__file__).parent.parent / "models"

ridge = joblib.load(MODEL_DIR / "best_baseline_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")

#input schema
class HouseFeatures(BaseModel):
    OverallQual:  int   = Field(..., ge=1, le=10, description="Overall quality 1-10")
    GrLivArea:    float = Field(..., gt=0,        description="Above grade living area sqft")
    GarageCars:   int   = Field(..., ge=0, le=5,  description="Garage capacity in cars")
    TotalBsmtSF:  float = Field(..., ge=0,        description="Total basement sqft")
    FullBath:     int   = Field(..., ge=0,        description="Full bathrooms above grade")
    HalfBath:     int   = Field(0,  ge=0,        description="Half bathrooms above grade")
    YearBuilt:    int   = Field(..., ge=1800, le=2025, description="Year built")
    YearRemodAdd: int   = Field(..., ge=1800, le=2025, description="Year remodeled")
    LotArea:      float = Field(..., gt=0,        description="Lot size sqft")
    ExterQual:    int   = Field(..., ge=0, le=5,  description="Exterior quality 0-5")
    GarageArea:   float = Field(0,  ge=0,        description="Garage area sqft")
    Fireplaces:   int   = Field(0,  ge=0,        description="Number of fireplaces")
    Neighborhood: str   = Field(...,             description="Neighborhood name")


#output schema
class PredictionResponse(BaseModel):
    predicted_price:      float
    predicted_price_low:  float
    predicted_price_high: float
    model_rmse:           float

#helper building feature rows
def build_feature_row(data: HouseFeatures) -> pd.DataFrame:
    d = data.dict()
 
    # derived features — same as Day 3
    d["TotalSF"]      = d["TotalBsmtSF"]
    d["HouseAge"]     = 2010 - d["YearBuilt"]
    d["WasRemodeled"] = int(d["YearRemodAdd"] != d["YearBuilt"])
    d["RemodAge"]     = 2010 - d["YearRemodAdd"]
    d["TotalBath"]    = d["FullBath"] + d["HalfBath"] * 0.5
    d["HasGarage"]    = int(d["GarageArea"] > 0)
    d["HasBasement"]  = int(d["TotalBsmtSF"] > 0)
    d["HasFireplace"] = int(d["Fireplaces"] > 0)
    d["HasPool"]      = 0
 
    # handle neighborhood one-hot encoding
    neighborhood_col = f"Neighborhood_{d['Neighborhood']}"
    del d["Neighborhood"]
 
    row = pd.DataFrame([d])
 
    # add neighborhood column if it exists in training features
    if neighborhood_col in feature_cols:
        row[neighborhood_col] = 1
 
    # reindex to match training columns exactly
    row = row.reindex(columns=feature_cols, fill_value=0)
 
    return row
 

#routes
app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs":    "/docs",
        "health":  "/health"
    }
 
 
@app.get("/health")
def health():
    return {"status": "ok", "model": "Ridge + XGBoost blend", "rmse": 0.0694}
 
 
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        row = build_feature_row(features)
 
        row_scaled = scaler.transform(row)
        print("\n===== SCALED VALUES =====")
        print(row_scaled[:, :10])
        
        print("\n===== INPUT ROW =====")
        print(row.T)

        print("\n===== SHAPE =====")
        print(row.shape)

        print("\n===== FIRST 20 FEATURES =====")
        print(row.iloc[:, :20])
        ridge_pred = ridge.predict(row_scaled)[0]
        xgb_pred = xgb_model.predict(row)[0]

        # Debug prints
        print("Ridge prediction:", ridge_pred)
        print("XGBoost prediction:", xgb_pred)

        blended_log = (ridge_pred + xgb_pred) / 2

        print("Blended prediction:", blended_log)
        print("Price after expm1:", np.expm1(blended_log))

        price = float(np.expm1(blended_log))
        price_low = float(np.expm1(blended_log - 0.0694))
        price_high = float(np.expm1(blended_log + 0.0694))
        price_low    = float(np.expm1(blended_log - 0.0694))
        price_high   = float(np.expm1(blended_log + 0.0694))
 
       
        return PredictionResponse(
            predicted_price=      round(price, 2),
            predicted_price_low=  round(price_low, 2),
            predicted_price_high= round(price_high, 2),
            model_rmse=           0.0694
         )
 
        
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
#Run
 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)