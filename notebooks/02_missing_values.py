import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

df = pd.read_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train.csv") 

#missing vals
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

print("Columns with missing values and their percentage:\n", missing_pct)

#NaN columns for none
none_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
df[none_cols] = df[none_cols].fillna("None")

print("Missing values after filling 'None':\n", df[none_cols].isnull().sum())

#NaN columns for 0
zero_cols = [
    "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]
df[zero_cols] = df[zero_cols].fillna(0)

print("Missing values after filling 0:\n", df[zero_cols].isnull().sum())

#LotFrontage
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#remaining categorical columns
mode_cols = [
    "MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType",
    "Functional", "Utilities"
]
df[mode_cols] = df[mode_cols].fillna(df[mode_cols].mode().iloc[0])

#verify
print(f"Total missing values after processing: {df.isnull().sum().sum()}")

#visualization 
plt.figure(figsize=(10, 6))
missing_pct.head(20).plot(kind='barh', color='skyblue')
plt.title("Top 20 Columns with Missing Values")
plt.xlabel("Percentage of Missing Values")
plt.axvline(x=30, color='red', linestyle='--',label='30% Threshold')
plt.legend()
plt.tight_layout()
plt.show()

#saving
df.to_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train_cleaned.csv", index=False)
print("Cleaned dataset saved successfully.",df.shape)