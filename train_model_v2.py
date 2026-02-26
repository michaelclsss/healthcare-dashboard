import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
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
    X = df.drop(columns=['readmitted_binary'])
    y = df['readmitted_binary']
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts()}")
    return X, y

def train_model(X_train, y_train):
    """Train a Gradient Boosting model without SMOTE, using class weights to handle the imbalance."""
    print("\nTraining Gradient Boosting model (no SMOTE)...")

    # Calculate how imbalanced the classes are
    # If there are 8x more non-readmitted patients, tell the model to treat each
    # readmitted patient as 8x more important
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale = n_negative / n_positive
    print(f"Class weight scale: {scale:.1f}x (readmitted patients weighted higher)")

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )

    # Pass sample weights directly — each readmitted patient counts
    # 8x more than a non-readmitted patient during training
    sample_weights = y_train.map({0: 1.0, 1: scale})
    # To compensate for not using SMOTE, we create sample weights that give more importance to 
    # the minority readamitted class
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """This runs the trained model against the 20% of patients it has never seen before."""
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test) # Models best guess, 0 or 1, readmitted or not
    y_prob = model.predict_proba(X_test)[:, 1] # gives a probability instead — like "this patient has a 7
                                                # 1% chance of readmission."

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Readmitted', 'Readmitted <30 days']))

    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {auc:.3f}")
    print("(0.5 = random guessing, 1.0 = perfect, 0.75+ = good)")

    return y_pred, y_prob, auc

def plot_confusion_matrix(y_test, y_pred):
    """Plot a confusion matrix to see model performance"""
    os.makedirs('plots', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.title('Confusion Matrix — Gradient Boosting v3', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_v3.png', dpi=150)
    plt.close()
    print("Saved confusion_matrix_v3.png")

def plot_feature_importance(model, X):
    """Show which features made the biggest differnce to the model predictions"""
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature',
                hue='feature', palette='viridis', legend=False)
    plt.title('Top 15 Features — Gradient Boosting v3', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/feature_importance_v3.png', dpi=150)
    plt.close()
    print("Saved feature_importance_v3.png")

def plot_roc_curve(y_test, y_prob, auc):
    """Create a ROC curve to show tradeoff between differnet threshholds"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Gradient Boosting v3', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/roc_curve_v3.png', dpi=150)
    plt.close()
    print("Saved roc_curve_v3.png")

def save_model(model):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/readmission_model_v3.pkl')
    print("\nModel saved to models/readmission_model_v3.pkl")

def run_training():
    df = load_cleaned_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    model = train_model(X_train, y_train)
    y_pred, y_prob, auc = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X)
    plot_roc_curve(y_test, y_prob, auc)
    save_model(model)

    return model, X_test, y_test

if __name__ == "__main__":
    run_training()


