# ==========================================
# CUSTOMER CHURN DATA PREPARATION & EDA
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# ------------------------------------------
# 1. DATA UNDERSTANDING
# ------------------------------------------

# Load Dataset
df = pd.read_csv("C:\\Users\\ksche\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe(include='all'))

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Check Blank Spaces
print("\nRows with Blank TotalCharges")
print((df['TotalCharges'] == " ").sum())


# ------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------

# Replace blank spaces with NaN
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)

# Convert TotalCharges to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Check missing values
print("\nMissing Values After Conversion")
print(df.isnull().sum())

# Fill missing values with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Verify datatype
print(df['TotalCharges'].dtype)

# Remove customerID (not useful for modeling)
df.drop('customerID', axis=1, inplace=True)

print("\nCleaned Dataset Shape:", df.shape)


# ------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------

# Tenure Group

def tenure_group(tenure):
    if tenure <= 12:
        return "0-12 Months"
    elif tenure <= 24:
        return "13-24 Months"
    elif tenure <= 48:
        return "25-48 Months"
    elif tenure <= 60:
        return "49-60 Months"
    else:
        return "60+ Months"

df['TenureGroup'] = df['tenure'].apply(tenure_group)

# Average Monthly Spend
df['AvgMonthlySpend'] = np.where(
    df['tenure'] == 0,
    df['MonthlyCharges'],
    df['TotalCharges'] / df['tenure']
)

# Convert Yes/No Columns to 1/0

binary_cols = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'Churn'
]

for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

# Gender Encoding
df['gender'] = df['gender'].map({'Male':1, 'Female':0})

# One-Hot Encoding

categorical_cols = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
    'TenureGroup'
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nEncoded Dataset Shape:", df_encoded.shape)


# ------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------

sns.set_style("whitegrid")

# ------------------------------------------
# Countplot for Churn
# ------------------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()

# ------------------------------------------
# Contract vs Churn
# ------------------------------------------

plt.figure(figsize=(8,5))
sns.countplot(
    x='Contract',
    hue='Churn',
    data=df
)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# ------------------------------------------
# Monthly Charges vs Churn
# ------------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(
    x='Churn',
    y='MonthlyCharges',
    data=df
)
plt.title("Monthly Charges by Churn")
plt.show()

# ------------------------------------------
# Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(18,10))

corr_matrix = df_encoded.corr()

sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# Pairplot
# ------------------------------------------

pairplot_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'Churn'
]

sns.pairplot(
    df[pairplot_cols],
    hue='Churn'
)

plt.show()


# ------------------------------------------
# TOP CORRELATIONS WITH CHURN
# ------------------------------------------

churn_corr = df_encoded.corr()['Churn']

top_corr = churn_corr.sort_values(
    ascending=False
)

print("\nTop Positive Correlations with Churn")
print(top_corr.head(10))

print("\nTop Negative Correlations with Churn")
print(top_corr.tail(10))


# ------------------------------------------
# SAVE CLEANED DATA
# ------------------------------------------

df_encoded.to_csv(
    "Customer_Churn_Cleaned.csv",
    index=False
)

print("\nCleaned CSV Saved Successfully")