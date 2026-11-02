"""
Title: Healthcare Classification Lifecycle
Description: Module 7 Supervised Segmentation & Classification Exercise.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, LogisticRegression from sklearn.linear_model, accuracy_score, and confusion_matrix
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup 
    - Goal: Load 'patient_classification_data.csv' and inspect class distribution.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./patient_classification_data.csv")
    # TODO: Check class balance of 'High_Risk_Hypertension' (value_counts)
    counts = df["High_Risk_Hypertension"].value_counts()
    print(counts)
    return df


def phase_2_supervised_segmentation(df):
    """
    [Phase 2] Supervised Segmentation & Feature Selection 
    - Goal: Analyze feature differences between risk groups.
    """
    print("\n[PHASE 2] Supervised Segmentation")
    print("-" * 65)
    # TODO: Group data by 'High_Risk_Hypertension' and calculate mean of numeric features (BMI, Daily_Steps)
    grouped_data = df.groupby("High_Risk_Hypertension")[["BMI", "Daily_Steps"]].mean()
    print(grouped_data)
    return df


def phase_3_classification_modeling(df):
    """
    [Phase 3] Classification Model Induction 
    - Goal: Train a Logistic Regression model to classify high-risk patients.
    """
    print("\n[PHASE 3] Classification Modeling")
    print("-" * 65)
    # TODO: Define features X = df[['BMI', 'Daily_Steps']] and target y = df['High_Risk_Hypertension']
    X = df[['BMI', 'Daily_Steps']]
    y = df['High_Risk_Hypertension']
    # TODO: Fit LogisticRegression model
    model = LogisticRegression()
    model = model.fit(X,y)
    return model,X,y,df


def phase_4_model_evaluation(model, X, y):
    """
    [Phase 4] Evaluating Classifiers & Generalization 
    - Goal: Compute Accuracy and Confusion Matrix.
    - Provide clinical prescriptive actions.
    """
    print("\n[PHASE 4] Model Evaluation & Clinical Action")
    print("-" * 65)
    y_pred = model.predict(X)
    # TODO: Generate predictions and calculate accuracy_score and confusion_matrix
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(cm)
    # TODO: Print prescriptive clinical intervention recommendations
    if accuracy >= 0.90:
        print("Model shows good classification performance.")
        print("Continue regular clinical monitoring.")

    elif accuracy >= 0.70:
        print("Model shows moderate classification performance.")
        print("Review high-risk patient predictions carefully.")

    else:
        print("Model shows poor classification performance.")
        print("Improve feature selection and model tuning.")

    print("Recommend clinical assessment for patients classified as high risk.")
    pass


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING HEALTHCARE CLASSIFICATION LIFECYCLE ")
    print("=" * 65)
    
    df = phase_1_discovery_and_setup()
    df = phase_2_supervised_segmentation(df)
    model, X, y, df = phase_3_classification_modeling(df)
    phase_4_model_evaluation(model, X, y)


if __name__ == "__main__":
    run_analytics_lifecycle()