import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
#data loading
df = pd.read_csv("A:\\AIML_Projects\\House_Price_Prediction\\data\\train.csv") 
print("Shape of the dataset:", df.shape)
df.dtypes.value_counts()
print(df.head())

#missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_count = missing_values[missing_values > 0]
missing_percentage = ((missing_count / len(df)) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing_count, 
    "Missing Percentage": missing_percentage})

print(missing_df)

#SalePrice
# raw SalePrice
plt.figure(figsize=(6, 4))
sns.histplot(df["SalePrice"], kde=True)
plt.title("SalePrice — Raw")
plt.show()

# log-transformed SalePrice
plt.figure(figsize=(6, 4))
sns.histplot(np.log1p(df["SalePrice"]), kde=True)
plt.title("SalePrice — Log Transformed")
plt.show()

#correlation
corr = df.select_dtypes(include=[np.number]).corr()["SalePrice"]

top_corr =corr.abs().sort_values(ascending=False).head(15)

sns.barplot(x=top_corr.values, y=top_corr.index)
#scatter
plt.figure(figsize=(8, 5))

sns.scatterplot(x='GrLivArea', y='SalePrice', data=df,hue='OverallQual', palette='viridis',alpha=0.6)  


plt.tight_layout()
plt.show()

#boxplot1
plt.figure(figsize=(10, 5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df,hue='OverallQual', palette='Blues')


plt.tight_layout()
plt.show()

#boxplot2
plt.figure(figsize=(8, 5))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df, hue='Neighborhood', palette='Blues')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#observations

observations = """ 
1. The dataset contains 1460 rows and 81 columns, with various data types including integers, floats, and objects.
2. There are several missing values in the dataset, with the most significant being 'PoolQC', 'MiscFeature', and 'Alley'.
3. The 'SalePrice' variable is right-skewed, but log transformation helps in normalizing its distribution.
4. The top features correlated with 'SalePrice' include 'OverallQual', 'GrLivArea', 'GarageCars', and 'TotalBsmtSF'.
5. Scatter plots indicate a positive relationship between 'GrLivArea' and 'SalePrice', with 'OverallQual' acting as a significant factor in this relationship.
6.Spotted some outliers in 'GrLivArea' and 'SalePrice',large houses with unusually low sale prices, wlikely non-market sales
"""
print(observations)
