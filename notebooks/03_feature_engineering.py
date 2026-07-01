import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid")

df = pd.read_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train_cleaned.csv")

print("Shape of the dataset:", df.shape)

#dealing with outliers
df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000))]
print("Shape of the dataset:", df.shape)

#new features
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] #total floor 
df["HouseAge"] = df["YrSold"] - df["YearBuilt"] #house age
df["WasRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int) #was remodeled
df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"] #remodel age
df["TotalBath"] = df["FullBath"] + (0.5 * df["HalfBath"]) + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]) #total bathrooms
df["HasPool"] = (df["PoolArea"] > 0).astype(int) #has pool
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

print("New features that are added successfully:", df[["TotalSF", "HouseAge", "WasRemodeled", "RemodelAge", "TotalBath", "HasPool", "HasGarage", "HasFireplace", "HasBasement"]].head())

#Log transformation 
df["SalePriceLog"] = np.log1p(df["SalePrice"])

#skew fixation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["Id", "SalePrice", "SalePriceLog"]]

skewness = df[numeric_cols].skew().sort_values(ascending=False)
skewed_cols = skewness[abs(skewness) > 0.75].index.tolist()
df[skewed_cols] = np.log1p(df[skewed_cols])

print(f"{len(skewed_cols)} skewed numeric columns have been log-transformed")

#ordinal categorical columns
quality_map = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}

ordinal_cols = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC"
]

# Fill missing values with "None" first, then map
for col in ordinal_cols:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None").map(quality_map)

# Other ordinal columns

if df["BsmtExposure"].dtype == "object":
    df["BsmtExposure"] = (
        df["BsmtExposure"]
        .fillna("None")
        .map({
            "None": 0,
            "No": 1,
            "Mn": 2,
            "Av": 3,
            "Gd": 4
        })
    )

if df["BsmtFinType1"].dtype == "object":
    df["BsmtFinType1"] = (
        df["BsmtFinType1"]
        .fillna("None")
        .map({
            "None": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6
        })
    )

if df["BsmtFinType2"].dtype == "object":
    df["BsmtFinType2"] = (
        df["BsmtFinType2"]
        .fillna("None")
        .map({
            "None": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6
        })
    )

if df["GarageFinish"].dtype == "object":
    df["GarageFinish"] = (
        df["GarageFinish"]
        .fillna("None")
        .map({
            "None": 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3
        })
    )

if df["Functional"].dtype == "object":
    df["Functional"] = df["Functional"].map({
        "Sal": 1,
        "Sev": 2,
        "Maj2": 3,
        "Maj1": 4,
        "Mod": 5,
        "Min2": 6,
        "Min1": 7,
        "Typ": 8
    })

if df["LandSlope"].dtype == "object":
    df["LandSlope"] = df["LandSlope"].map({
        "Sev": 1,
        "Mod": 2,
        "Gtl": 3
    })

if df["PavedDrive"].dtype == "object":
    df["PavedDrive"] = df["PavedDrive"].map({
        "N": 0,
        "P": 1,
        "Y": 2
    })

# Verify
all_ordinal = ordinal_cols + [
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageFinish",
    "Functional",
    "LandSlope",
    "PavedDrive"
]

print("\nMissing values after ordinal encoding:")
print(df[all_ordinal].isnull().sum())

print("\nTotal missing values in dataset:")
print(df.isnull().sum().sum())

missing = df.isnull().sum()
missing = missing[missing > 0]

print(missing)

#one hot encode nominal categorical columns
nominal_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Nominal categorical columns to be one-hot encoded: {nominal_cols}")
print(nominal_cols)

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
print(f"Shape of the dataset after one-hot encoding: {df.shape}")

#final check
print(df.isnull().sum().sum())

print("Final Shape:", df.shape)

#saving
df.to_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train_final.csv", index=False)

# print(df["PoolQC"].value_counts(dropna=False))
# # check what values exist vs what's in your map
# for col in ordinal_cols:
#     unique_vals = set(df[col].unique())
#     map_keys = set(quality_map.keys())
#     unmapped = unique_vals - map_keys
#     if unmapped:
#         print(f"{col} has unmapped values: {unmapped}")