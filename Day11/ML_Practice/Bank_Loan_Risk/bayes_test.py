"""
Title: Supervised Classification Pipeline for Loan Default Risk 
Description: A modular Python exercise script for loading clean loan applicant data, 
             identifying target/predictors, encoding categorical variables, 
             scaling features, training a Logistic Regression classifier, evaluating performance, 
             and persisting the trained pipeline artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, train_test_split, StandardScaler, 
#       LogisticRegression, accuracy_score, classification_report, confusion_matrix, and joblib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery, Setup & Target/Predictor Identification
    - Goal: Load 'loan_risk_data.csv' and inspect structure.
    - Drop non-predictive identifiers (e.g., 'applicant_id').
    - Identify Target feature (y: 'default_flag') and Predictor features (X).
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv(r"D:\Machine_Learning(cdac)\my_practise\Day11\ML_Practice\Bank_Loan_Risk\loan_risk_data.csv")
    # TODO: Separate target feature y ('default_flag') and predictor features X
    X = df.drop(columns='default_flag')
    y = df['default_flag']
    # TODO: Drop ID column ('applicant_id')
    X = X.drop(columns='applicant_id')
    # TODO: Inspect class distribution of target variable
    print(y.value_counts())
    return X,y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Categorical Encoding & Feature Engineering
    - Goal: Encode categorical variables (e.g., 'housing_status').
    - Optionally create engineered features (e.g., loan-to-income ratio).
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    # TODO: Apply one-hot encoding to categorical columns using pd.get_dummies()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    X_encoded = pd.get_dummies(X,columns=cat_cols,drop_first=True,dtype=int)
    # TODO: Add engineered features if desired (e.g., X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k'])
    X_encoded['loan_to_income_ratio'] = X_encoded['loan_amount_k'] / X_encoded['annual_income_k']
    # TODO: Display head of transformed feature matrix
    print(X.head())
    return X_encoded


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Train-Test Split & Feature Scaling
    - Goal: Split dataset into Training and Testing sets.
    - Fit StandardScaler on training data and transform both train and test sets.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    # TODO: Perform train_test_split (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
    # TODO: Initialize StandardScaler and scale numerical features
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=[np.number]).columns
    # TODO: Fit scaler on X_train, transform X_train and X_test
    X_train_encoded = scaler.fit_transform(X_train[num_cols])
    X_test_encoded = scaler.transform(X_test[num_cols])
    return X_train_encoded, X_test_encoded, y_train, y_test, scaler


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Model Training & Classification Metrics
    - Goal: Train a LogisticRegression model.
    - Predict on test set and calculate evaluation metrics (Accuracy, Confusion Matrix, Classification Report).
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    # TODO: Instantiate LogisticRegression(random_state=42) and fit on training data
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    # TODO: Generate predictions on test set
    y_pred = model.predict(X_test_scaled)
    print(y_pred)
    # TODO: Compute accuracy_score and classification_report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(accuracy)
    print(report)
    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Model Persistence & Future Deployment Setup
    - Goal: Save the trained model, fitted scaler, and feature list using joblib.
    - Test loading saved artifacts to run an inference on a new applicant.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    model_dir = Path('./model')
    model_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Use joblib.dump() to save the model and scaler to disk (.joblib files)
    # joblib.dump(model, model_dir / "model.joblib")
    # joblib.dump(scaler, model_dir / "scaler.joblib")
    # joblib.dump(feature_names, model_dir / "features.joblib")
    # TODO: Load saved artifacts using joblib.load() and perform sample inference
    loaded_model = joblib.load(model_dir / "model.joblib")
    loded_scaler = joblib.load(model_dir / "scaler.joblib")
    # feature = joblib.load(model_dir / "features.joblib")
    
    print(loaded_model)
    print(loded_scaler)
    print(feature)
    return loaded_model, loded_scaler, feature


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_ml_pipeline():
    """
    Facade function coordinating the entire loan risk classification pipeline.
    """
    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE (Logistic Regression)")
    print("=" * 65)
    
    # Executing phases chronologically
    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    # phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_ml_pipeline()