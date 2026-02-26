import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_cleaned_data():
    print("Loading cleaned data...")
    df = pd.read_csv('data/cleaned_data.csv')
    print(f"Loaded {df.shape[0]:,} rows")
    return df

def split_features_target(df):
    """Separate the thing we're predicting from the features used to predict it"""
    X = df.drop(columns=['readmitted_binary']) # x = features, y = target answer, y or n
    y = df['readmitted_binary'] # x = age, num of medications...
    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts()}")
    return X, y

def handle_imbalance(X_train, y_train):
    """
    Only ~11% of patients are readmitted within 30 days.
    If we don't handle this, the model learns to just say 'not readmitted' beacuse we 
    have binary answer and still gets 89% accuracy — which is useless.
    SMOTE creates synthetic examples of the minority class to balance things out.
    """
    print("\nHandling class imbalance with SMOTE...")
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,    # 100 decision trees
        max_depth=10,        # limit tree depth to avoid overfitting
        min_samples_split=10,  # need at least 10 patients to make a split
        random_state=42,     # makes results reproducible
        n_jobs=-1           # use all CPU cores to train faster
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """Run the model on patients it has never seen before"""
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of readmission

    # Overall accuracy
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Readmitted', 'Readmitted <30 days']))

    # AUC-ROC score (better metric than accuracy for imbalanced data)
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {auc:.3f}")
    print("(0.5 = random guessing, 1.0 = perfect, 0.75+ = good)")

    return y_pred, y_prob, auc

def plot_confusion_matrix(y_test, y_pred):
    """Visual breakdown of correct vs incorrect predictions"""
    os.makedirs('plots', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.title('Confusion Matrix — Readmission Prediction', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

def plot_feature_importance(model, X):
    """Shows which features matter most to the model"""
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Features for Predicting Readmission', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()
    print("Saved feature_importance.png")

def plot_roc_curve(y_test, y_prob, auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2, 
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Readmission Risk Model', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=150)
    plt.close()
    print("Saved roc_curve.png")

def save_model(model, scaler=None):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/readmission_model.pkl')
    print("\nModel saved to models/readmission_model.pkl")

def run_training():
    # Load data
    df = load_cleaned_data()
    X, y = split_features_target(df)

    # Split into training set (80%) and test set (20%)
    # The test set is patients the model never sees during training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    # Handle imbalance, train, evaluate
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    model = train_model(X_train_balanced, y_train_balanced)
    y_pred, y_prob, auc = evaluate_model(model, X_test, y_test)

    # Generate plots
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X)
    plot_roc_curve(y_test, y_prob, auc)

    # Save model for use in the dashboard
    save_model(model)

    return model, X_test, y_test

if __name__ == "__main__":
    run_training()
