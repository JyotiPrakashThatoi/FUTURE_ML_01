import pandas as pd
import numpy as np
import os

# Load the dataset
data_path = "data/superstore_sales.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path, encoding='latin1')

print("--- Initial Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Basic Cleaning
# Convert date columns to datetime
date_columns = ['Order Date', 'Ship Date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='mixed')

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicates: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)

# Sort by Order Date
if 'Order Date' in df.columns:
    df.sort_values('Order Date', inplace=True)

# Save cleaned data
cleaned_path = "data/cleaned_sales.csv"
df.to_csv(cleaned_path, index=False)
print(f"\nCleaned data saved to {cleaned_path}")

# Quick EDA Summary
if 'Sales' in df.columns:
    print("\n--- Sales Summary ---")
    print(df['Sales'].describe())
