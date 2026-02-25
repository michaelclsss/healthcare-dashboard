# Explore the csv file and print out some easy statistics.

import pandas as pd

df = pd.read_csv('data/diabetic_data.csv')

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nReadmitted column values:\n", df['readmitted'].value_counts())