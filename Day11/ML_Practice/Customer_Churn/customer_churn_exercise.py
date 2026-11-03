"""
Title: Supervised Classification Pipeline with Random Forest 
Description: A modular Python exercise script for loading clean customer churn data, 
             identifying target/predictors, encoding categorical variables, 
             scaling features, training a Random Forest classifier, evaluating performance, 
             and persisting the trained pipeline artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, train_test_split, StandardScaler, 
#       RandomForestClassifier, accuracy_score, classification_report, confusion_matrix, and joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib


def phase_1_discovery_and_target_identification():
    """
    [Phase 1] Discovery, Setup & Target/Predictor Identification
    - Goal: Load 'customer_churn_data.csv' and inspect structure.
    - Drop non-predictive identifiers (e.g., 'customer_id').
    - Identify Target feature (y: 'churn_flag') and Predictor features (X).
    """
    print("\n[PHASE 1] Discovery & Target Identification")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./customer_churn_data.csv")
    # TODO: Separate target feature y ('churn_flag') and predictor features X
    X = df.drop(columns='churn_flag')
    y = df["churn_flag"]
    # TODO: Drop ID column ('customer_id')
    X_clean = X.drop(columns='customer_id')
    print(X_clean.head())
    # TODO: Inspect class distribution of target variable
    frequency = y.value_counts().sort_index()
    print(frequency)
    return X_clean, y


def phase_2_feature_engineering_and_encoding(X):
    """
    [Phase 2] Categorical Encoding & Feature Engineering
    - Goal: Encode categorical variables (e.g., 'contract_type', 'internet_service').
    - Optionally create engineered interaction features (e.g., charges per tenure month).
    """
    print("\n[PHASE 2] Feature Engineering & Categorical Encoding")
    print("-" * 65)
    # TODO: Apply one-hot encoding to categorical columns using pd.get_dummies()
    # print(X.dtypes)
    # print(X.head())
    cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    X_ecoded = pd.get_dummies(X, columns=cat_cols, drop_first=True,dtype=int)
    # print(X_ecoded.head())
    
    #Using One Hot Encoder
    # encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
    # encoded_data = encoder.fit_transform(X[cat_cols])
    # encoded_cols = encoder.get_feature_names_out(cat_cols)
    # print(encoded_cols)
    # num_cols = X.select_dtypes(include=np.number).columns
    
    # encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=X.index)
    # print(encoded_df)
    
    # X_num = X[num_cols]
    # print(X_num)
    
    # X_final = pd.concat([X_num, encoded_df], axis=1)
    
    # TODO: Add engineered features if desired (e.g., X['avg_monthly_spend'] = X['total_charges'] / X['tenure_months'])
    X_ecoded["avg_monthly_spend"] = X_ecoded['total_charges'] / X_ecoded['tenure_months']
    # TODO: Display head of transformed feature matrix
    # print(X_ecoded.head())
    
    return X_ecoded


def phase_3_scaling_and_train_test_split(X, y):
    """
    [Phase 3] Train-Test Split & Feature Scaling
    - Goal: Split dataset into Training and Testing sets.
    - Fit StandardScaler on training data and transform both train and test sets.
    """
    print("\n[PHASE 3] Data Splitting & Feature Scaling")
    print("-" * 65)
    # TODO: Perform train_test_split (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)
    # TODO: Initialize StandardScaler and scale numerical features
    print(X.dtypes)
    scalar = StandardScaler()
    # TODO: Fit scaler on X_train, transform X_train and X_test
    X_train_encoded = scalar.fit_transform(X_train)
    X_test_encoded = scalar.transform(X_test)
    
    return X_train_encoded,X_test_encoded,y_train,y_test,scalar


def phase_4_model_training_and_testing(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    [Phase 4] Random Forest Training & Classification Metrics
    - Goal: Train a RandomForestClassifier model.
    - Predict on test set and calculate evaluation metrics (Accuracy, Precision, Recall, F1).
    """
    print("\n[PHASE 4] Model Training & Test Set Evaluation")
    print("-" * 65)
    # TODO: Instantiate RandomForestClassifier(random_state=42) and fit on training data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    # TODO: Generate predictions on test set
    y_pred = model.predict(X_test_scaled)
    # TODO: Compute accuracy_score and classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred)
    print("Report: ", report)
    return model


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Model Persistence & Future Deployment Setup
    - Goal: Save the trained model, fitted scaler, and feature list using joblib.
    - Test loading saved artifacts to run an inference on a new sample.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    model_dir = Path("./model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    
    # TODO: Use joblib.dump() to save the model and scaler to disk (.joblib files)
    joblib.dump(model, model_dir / "customer_churn.joblib")
    joblib.dump(scaler, model_dir / "scalere.joblib")
    joblib.dump(feature_names, model_dir / "features.joblib")
    
    # TODO: Load saved artifacts using joblib.load() and perform sample inference
    loaded_model = joblib.load(model_dir / "customer_churn.joblib")
    loaded_scaler = joblib.load(model_dir / "scalere.joblib")
    loader_feature = joblib.load(model_dir / "features.joblib")
    
    
    print("Model: \n", loaded_model)
    print("scaler: \n", loaded_scaler)
    print("Feature: \n", loader_feature)
    return loaded_model, loaded_scaler, loader_feature


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_ml_pipeline():
    """
    Facade function coordinating the entire classification pipeline.
    """
    print("=" * 65)
    print("STARTING SUPERVISED CLASSIFICATION LIFECYCLE (Random Forest)")
    print("=" * 65)
    
    # Executing phases chronologically
    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_ml_pipeline()