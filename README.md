# House Price Prediction

> End-to-end machine learning project predicting residential home sale prices using the Ames Housing dataset. Built with XGBoost, FastAPI, and Docker.

---

## Problem Statement

Predict the sale price of a house given 79 features describing its size, quality, location, and condition. The goal is to build a model accurate enough to be useful in a real estate context, and deploy it as a production-ready REST API.
![API Demo](assets/api_demo.png)

---

## Dataset

- **Source:** [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques):- Kaggle
- **Size:** 1,460 training samples · 79 features
- **Target:** `SalePrice` (log-transformed for modeling)

---

## Approach

### Step 1:EDA
- Identified right-skewed target distribution → applied log1p transform
- Found 19 columns with missing values: audited patterns
- Discovered 2 outlier houses (large area, abnormally low price) → non-market sales
- Key finding: `OverallQual`, `GrLivArea`, `GarageCars` are strongest predictors

### Step 2: Missing Value Treatment
- **"None" fill** for 15 categorical columns where NaN = feature absent (e.g. `PoolQC`, `Alley`)
- **Zero fill** for 10 numeric columns where NaN = 0 (e.g. `GarageArea`, `MasVnrArea`)
- **Neighborhood median** imputation for `LotFrontage`: grouped by location
- Result: **0 missing values** across all 1,460 rows

### Step 3: Feature Engineering
- Dropped 2 outlier rows → 1,458 rows remaining
- Created 9 derived features:
  - `TotalSF`: combined basement + floor areas
  - `HouseAge`: age at time of sale
  - `WasRemodeled`: binary remodel flag
  - `RemodAge`: years since remodel
  - `TotalBath`: weighted bathroom count
  - `HasPool`, `HasGarage`, `HasBasement`, `HasFireplace`
- Log-transformed 25 skewed numeric features
- Ordinal encoded 17 quality/condition columns
- One-hot encoded 26 nominal columns
- Result: **213 features** ready for modeling

### Step 4: Baseline Modeling
- Scaled features with `RobustScaler` (median + IQR, robust to outliers)
- 5-fold cross-validation
- Compared Linear Regression, Ridge, and Lasso
- **Best baseline: Ridge (alpha=10): CV RMSE: 0.1132**

### Step 5: XGBoost + Tuning
- RandomizedSearchCV over 50 parameter combinations
- Best params: `learning_rate=0.005`, `n_estimators=3000`, `max_depth=5`, `subsample=0.6`
- **XGBoost tuned CV RMSE: 0.1159**
- SHAP analysis for model explainability

---

## Results

| Model | CV RMSE | Notes |
|---|---|---|
| Ridge baseline | 0.1132 | Strong linear model on engineered features |
| XGBoost default | 0.1180 | Before tuning |
| XGBoost tuned | 0.1159 | After RandomizedSearchCV |

> **RMSE is in log-dollar units.** An RMSE of 0.1159 means predictions are typically within ~11.6% of the actual price. For a $200,000 house, that's ±$23,200.

> **Key insight:** Ridge outperformed XGBoost, suggesting the engineered features have strong linear relationships with price: a sign of effective feature engineering rather than model failure.

---

## Architecture

```
Input JSON
    ↓
FastAPI /predict endpoint
    ↓
Feature Engineering (replicate Step 3 pipeline)
    ↓
XGBoost Model
    ↓
np.expm1() → convert log prediction to dollars
    ↓
JSON Response {price, low, high, rmse}
```

---

## How to Run

### Option 1: Docker (recommended)

```bash
git clone https://github.com/BedantaMaharana/house-price-prediction.git
cd house-price-prediction

docker compose up --build

http://localhost:8000
http://localhost:8000/docs
```

### Option 2: Local

```bash
pip install -r requirements.txt

cd app
python main.py

http://localhost:8000/docs
```

---

## API Usage

### POST `/predict`

**Request:**
```json
{
  "OverallQual": 7,
  "GrLivArea": 1500,
  "GarageCars": 2,
  "TotalBsmtSF": 800,
  "FullBath": 2,
  "HalfBath": 1,
  "YearBuilt": 2000,
  "YearRemodAdd": 2005,
  "LotArea": 8000,
  "ExterQual": 3,
  "GarageArea": 400,
  "Fireplaces": 1,
  "Neighborhood": "CollgCr"
}
```

**Response:**
```json
{
  "predicted_price": 107044.46,
  "predicted_price_low": 95329.86,
  "predicted_price_high": 120198.59,
  "model_rmse": 0.1159
}
```

### GET `/health`
```json
{"status": "ok", "model": "XGBoost", "rmse": 0.1159}
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy, scipy |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Deployment | Docker, docker-compose |
| Language | Python 3.12 |

---

## Project Structure

```
house-price-prediction/
├── app/
│   ├── main.py              ← FastAPI routes
│   ├── model.py             ← model loading and prediction logic
│   └── schema.py            ← input/output schemas
├── notebooks/
│   ├── 01_eda.py            ← exploratory data analysis
│   ├── 02_missing_values.py ← missing value treatment
│   ├── 03_feature_engineering.py
│   ├── 04_baseline_model.py
│   ├── 05_xgboost.py        ← XGBoost tuning + SHAP
│   └── shap_analysis.py
├── models/                  ← saved model artifacts (gitignored)
├── data/                    ← raw and processed data (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
├── docs/
     ├── index.html          ← frontend
```

---

## Key Learnings

1. **Missing values encode information**: `PoolQC = NaN` means "no pool", not unknown. Treating them differently from genuinely missing values improved model quality.

2. **Feature engineering beat raw features**: `TotalSF` (combined area) and `HouseAge` consistently ranked as top predictors over the original separate columns.

3. **Strong features make linear models competitive**: Ridge's performance matching XGBoost shows that good feature engineering reduces the need for complex models.

4. **Preprocessing must be replicated at inference**: the API replicates every Step 3 transformation (log1p, derived features, encoding) to ensure predictions match training conditions.

---

##  Live Links
- **Website:** https:/BedantaMaharana.github.io/House_Price_Prediction/
- **API:** https://house-price-prediction-mk3c.onrender.com
- **API Docs:** https://house-price-prediction-mk3c.onrender.com/docs

## Contact

Built by [Bedanta Maharana] · [www.linkedin.com/in/bedanta-maharana-67b586317] · [[GitHub](https://github.com/BedantaMaharana)]