import shap
import joblib
import pandas as pd
import numpy as np

# load saved model and data
best_xgb = joblib.load("../models/xgboost_model.pkl")
df = pd.read_csv("../data/train_final.csv")

X = df.drop(columns=["Id", "SalePrice", "SalePriceLog"])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# rerun SHAP
explainer   = shap.Explainer(best_xgb)
shap_values = explainer(X)

# bar chart — feature importance
shap.summary_plot(shap_values, X, plot_type="bar", max_display=20)

# beeswarm — direction of impact
shap.summary_plot(shap_values, X, max_display=20)

# single house waterfall
shap.plots.waterfall(shap_values[0], max_display=15)