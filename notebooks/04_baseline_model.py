import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import warnings
import joblib
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

#loading
df = pd.read_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train_final.csv")
print("Loaded shape:",df.shape)

#split features
X = df.drop(["Id", "SalePrice", "SalePriceLog"], axis=1)
y = df["SalePriceLog"]

print("Feature(X) shape:", X.shape)
print("Target(y) shape:", y.shape)

#scaling feature
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

#cross validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_cv(model,X,y,cv):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return np.sqrt(-scores)

#LR base
lr = LinearRegression()
lr_scores = rmse_cv(lr, X_scaled, y, cv)
print(f"Linear Regression RMSE: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")

#ridge regression
alphas = [0.1, 1, 5, 10,50, 100]
ridge_scores = {}
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = rmse_cv(ridge, X_scaled, y, cv)
    ridge_scores[alpha] = scores.mean()
    print(f"Ridge Regression (alpha={alpha}) RMSE: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_alpha = min(ridge_scores, key=ridge_scores.get)
print(f"Best alpha for Ridge Regression: {best_alpha}")

#lasso regression
lasso_alphas = [0.0001,0.0005, 0.001,0.005, 0.01]

lasso_scores = {}
for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha)
    scores = rmse_cv(lasso, X_scaled, y, cv)
    lasso_scores[alpha] = scores.mean()
    print(f"Lasso Regression (alpha={alpha}) RMSE: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_alpha_lasso = min(lasso_scores, key=lasso_scores.get)
print(f"Best alpha for Lasso Regression: {best_alpha_lasso}")

#comparing
plt.figure(figsize=(8, 6))

models = ["Linear",f"Ridge(a = {best_alpha})",f"Lasso(a = {best_alpha_lasso})"]
scores = [rmse_cv(LinearRegression(), X_scaled, y, cv).mean(),
          rmse_cv(Ridge(alpha=best_alpha), X_scaled, y, cv).mean(),
          rmse_cv(Lasso(alpha=best_alpha_lasso,max_iter=10000), X_scaled, y, cv).mean()]

sns.barplot(x=models, y=scores,palette="Blues_d")
plt.title("Model Comparison (RMSE)[Lower the better]")
plt.ylabel("RMSE")
plt.ylim(min(scores) - 0.02, max(scores) + 0.02)
for i, s in enumerate(scores):
    plt.text(i, s + 0.005, f"{s:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()


#training best model
best_baseline = Ridge(alpha=best_alpha)
best_baseline.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "Feature": X.columns, 
    "Coefficient": best_baseline.coef_})

coef_df["abs_coef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="abs_coef", ascending=False)
print(coef_df.head(20))

#plotting top influential features
top20 = coef_df.head(20).sort_values(by="Coefficient", ascending=True)
colors = ['green' if c < 0 else 'red' for c in top20["Coefficient"]]
plt.figure(figsize=(10, 6))
plt.barh(top20["Feature"], top20["Coefficient"], color=colors)
plt.title("Top 20 Influential Features (Ridge Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

#saving results
model_path = "A:\\AIML_Projects\\House_Price_Prediction\\models"
joblib.dump(best_baseline, f"{model_path}\\best_baseline_model.pkl")
joblib.dump(scaler, f"{model_path}\\scaler.pkl")
joblib.dump(list(X.columns), f"{model_path}\\feature_columns.pkl")
joblib.dump(best_alpha, f"{model_path}\\best_alpha_ridge.pkl")

