# Implement the missing value, and outlier handling data preprocessing techniques on the dataset imported.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('attendance.csv')
# Handling Missing Values

# Option 1: Remove rows with missing values
df_cleaned = df.dropna()
# Option 2: Fill missing values with the mean (for numerical columns)
df_filled = df.fillna(df.select_dtypes(include=np.number).mean()) # Select only numeric columns for calculating the mean
# Option 3: Fill missing values with the median (for numerical columns)
df_filled_median = df.fillna(df.select_dtypes(include=np.number).median()) # Select numeric columns for median calculation
# Option 4: Fill missing values with the mode (for categorical columns)
df_filled_mode = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' else x)
# Handling Outliers

# Using the Interquartile Range (IQR) method
Q1 = df.select_dtypes(include=np.number).quantile(0.25) # Select numeric columns for quantile calculations
Q3 = df.select_dtypes(include=np.number).quantile(0.75) # Select numeric columns for quantile calculations
IQR = Q3 - Q1
# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Identify outliers
outliers = ((df.select_dtypes(include=np.number) < lower_bound) | (df.select_dtypes(include=np.number) > upper_bound))
# Option 1: Remove outliers
df_no_outliers = df[~((df.select_dtypes(include=np.number) < lower_bound) | (df.select_dtypes(include=np.number) > upper_bound)).any(axis=1)]
# Option 2: Cap outliers to the lower and upper bounds
df_capped = df.copy()
numeric_df = df.select_dtypes(include=np.number) # Select numeric columns for comparison

for col in numeric_df.columns:
    df_capped.loc[numeric_df[col] < lower_bound[col], col] = lower_bound[col]
    df_capped.loc[numeric_df[col] > upper_bound[col], col] = upper_bound[col]
# Option 3: Impute outliers with mean/median
df_imputed_outliers = df.copy()
df_imputed_outliers[outliers] = np.nan

# Calculate the mean only for numeric columns
numeric_df = df.select_dtypes(include=np.number)
df_imputed_outliers = df_imputed_outliers.fillna(numeric_df.mean())
print("Original DataFrame:\n", df)
print("DataFrame after handling missing values and outliers:\n", df_imputed_outliers)
