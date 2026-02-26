import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath):
    """Load raw CSV data"""
    print("Loading the data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean and handle missing values"""
    print("\nCleaning data...")

    # This dataset uses ? to represent missing values, replace it with NaN so pandas understands
    df = df.replace('?', np.nan)

    # Drop columns with too many missing values (>40% missing), this could result in bad training
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.4].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >40% missing: {cols_to_drop}")

    # Drop columns that won't help prediction
    drop_cols = ['encounter_id', 'patient_nbr', 'examide', 'citoglipton']
    drop_cols = [c for c in drop_cols if c in df.columns] # If its in the column, drop
    df = df.drop(columns=drop_cols)

    # Fill remaining missing values, guess and put in with median
    for col in df.columns:
        if df[col].dtype == 'object': # in pandas object is text/string
            df[col] = df[col].fillna(df[col].mode()[0])  # most common value
        else: # If object i snumeric
            df[col] = df[col].fillna(df[col].median())   # middle value

    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    return df

def engineer_features(df):
    """Create new useful features from existing columns"""
    print("\nEngineering features...")

    # Convert age ranges like "[70-80)" from data to a midpoint number for better modeling
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)

    # Count how many medications were changed (a useful risk signal)
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'insulin']
    med_cols = [c for c in med_cols if c in df.columns] # after dropping, check which columns remain
    df['num_med_changes'] = df[med_cols].apply(
        lambda row: sum(1 for val in row if val == 'Ch'), axis=1 # Add a column counting how many med were changed
    )

    # Total medications as a complexity score
    df['total_meds'] = df[med_cols].apply(
        lambda row: sum(1 for val in row if val != 'No'), axis=1 # Add column counting how many med patient is on
    )

    df['inpatient_ratio'] = df['number_inpatient'] / (
        df['number_outpatient'] + df['number_inpatient'] + df['number_emergency'] + 1
    )

    # Total number of visits across all types
    df['total_visits'] = (df['number_outpatient'] + 
                          df['number_inpatient'] + 
                          df['number_emergency'])

    # Flag for patients with high previous utilization
    df['high_utilizer'] = (df['number_inpatient'] > 2).astype(int)

    print(f"Added features: age (numeric), num_med_changes, total_meds, inpatient_ratio, total_visits, high_utilizer")

    return df

def encode_target(df):
    """Convert readmitted column to binary: 1 = readmitted <30 days, 0 = everything else"""
    print("\nEncoding target variable...")
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    print(f"Readmission rate: {df['readmitted_binary'].mean():.1%}")
    df = df.drop(columns=['readmitted'])
    return df

def encode_features(df):
    """Convert text columns to numbers using one-hot encoding"""
    print("\nEncoding categorical features...")
    
    # one-hot encode
    # (meaning they don't have too many unique values)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"Columns to encode: {cat_cols}")
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"Shape after encoding: {df.shape}")
    return df

def save_data(df, output_path):
    """Save cleaned data"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to {output_path}")
    print(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

def run_pipeline():
    # Run all steps in order
    df = load_data('data/diabetic_data.csv')
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_target(df)
    df = encode_features(df)
    save_data(df, 'data/cleaned_data.csv')
    return df

if __name__ == "__main__":
    run_pipeline()
