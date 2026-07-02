import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

#load dataa
df = pd.read_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train_final.csv")

X = df.drop(columns=['Id', 'SalePrice', 'SalePriceLog'])
y = df['SalePriceLog']

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert all columns to numeric and fill NaNs with 0

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

#cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model,X,y,cv):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

#XGBoost model with default parameters
xgb_base = xgb.XGBRegressor(n_estimators=1000,
                             random_state=42,
                             learning_rate=0.05,
                             max_depth=4,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             n_jobs=-1)

base_scores = rmse_cv(xgb_base, X, y, kf)
print(f"XGBoost default: {base_scores.mean():.4f} (+/- {base_scores.std() * 2:.4f})")

#Hyperparameter tuning 
param_grid = {
    'n_estimators': [1500,2000,3000],
    'learning_rate': [0.005, 0.01, 0.05],
    'max_depth': [3, 4, 5,6],
    'subsample': [0.6, 0.7,0.8,0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(xgb_model, 
                            param_distributions=param_grid, 
                            n_iter=25, 
                            scoring='neg_mean_squared_error', 
                            cv=kf, 
                            verbose=1, 
                            random_state=42,
                            n_jobs=-1)
search.fit(X, y)
print("Best parameters found: ", search.best_params_)
print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.4f}")
 
#train final model with best parameters

best_xgb = xgb.XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1,eval_metric='rmse')
best_xgb.fit(X, y)

#comparison of base and tuned model
tuned_scores = rmse_cv(best_xgb, X, y, kf)
print(f"XGBoost tuned — RMSE: {tuned_scores.mean():.4f} ± {tuned_scores.std():.4f}")

ridge_rmse = 0.1132

models = ["Ridge baseline", "XGBoost default", "XGBoost tuned"]
rmses  = [ridge_rmse, base_scores.mean(), tuned_scores.mean()]

plt.figure(figsize=(8, 4))
sns.barplot(x=models, y=rmses, palette="Blues_d")
plt.title("Model comparison — CV RMSE (lower is better)")
plt.ylabel("RMSE")
plt.ylim(min(rmses) - 0.02, max(rmses) + 0.02)
for i, r in enumerate(rmses):
    plt.text(i, r + 0.001, f"{r:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.show()

# summary
print("\n── Model Comparison Summary ──")
print(f"Ridge baseline : {ridge_rmse:.4f}")
print(f"XGBoost default: {base_scores.mean():.4f}")
print(f"XGBoost tuned  : {tuned_scores.mean():.4f}")
print(f"Improvement over Ridge: {((ridge_rmse - tuned_scores.mean()) / ridge_rmse * 100):.1f}%")

#BLEND 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

scaler    = RobustScaler()
X_scaled  = scaler.fit_transform(X)

ridge     = Ridge(alpha=10)
ridge.fit(X_scaled, y)

ridge_preds = ridge.predict(X_scaled)
xgb_preds   = best_xgb.predict(X)

blended      = (ridge_preds + xgb_preds) / 2
blended_rmse = np.sqrt(mean_squared_error(y, blended))
print(f"\nBlended RMSE (Ridge + XGBoost): {blended_rmse:.4f}")



#shap 
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, plot_type="bar", max_display=20)




#SHAP for single prediction
shap.plots.waterfall(shap_values[0], max_display=20)


#Residual analysis
preds = best_xgb.predict(X)
residuals = y - preds

plt.figure(figsize=(8, 4))
plt.scatter(preds, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted log(SalePrice)")
plt.ylabel("Residuals")
plt.title("Residual vs predicted")
plt.tight_layout()
plt.show()

joblib.dump(best_xgb, "A:\\AIML_Projects\\House_Price_Prediction\\models\\xgboost_model.pkl")
joblib.dump(explainer, "A:\\AIML_Projects\\House_Price_Prediction\\models\\xgboost_shap_explainer.pkl")
joblib.dump(list(X.columns), "A:\\AIML_Projects\\House_Price_Prediction\\models\\xgboost_feature_columns.pkl")
joblib.dump(shap_values, "A:\\AIML_Projects\\House_Price_Prediction\\models\\xgboost_shap_values.pkl")
