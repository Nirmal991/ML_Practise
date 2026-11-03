"""
Title: Unsupervised Customer Segmentation Pipeline with K-Means (Student Exercise)
Description: A modular Python exercise script for loading customer behavioral data, 
             scaling features for distance-based algorithms, training a K-Means clustering model, 
             evaluating cluster cohesion, profiling segments, and persisting artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, StandardScaler from sklearn.preprocessing, 
#       KMeans from sklearn.cluster, silhouette_score from sklearn.metrics, and joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import joblib

def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery, Setup & Feature Selection
    - Goal: Load 'customer_segmentation_data.csv' and inspect structure.
    - Drop non-analytical identifiers (e.g., 'customer_id').
    - Select numeric feature subset for clustering (e.g., annual income, spending score).
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./customer_segmentation_data.csv")
    # TODO: Drop ID column ('customer_id')
    df = df.drop(columns='customer_id')
    # TODO: Select relevant features for clustering (e.g., 'annual_income_k', 'spending_score_1_100', 'purchase_frequency_monthly')
    X = df[['annual_income_k', 'spending_score_1_100', 'purchase_frequency_monthly']]
    print(X.head())
    return X


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    - Goal: Initialize and fit StandardScaler on clustering features.
    - Unsupervised algorithms like K-Means rely heavily on Euclidean distance, making scaling mandatory.
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    # TODO: Initialize StandardScaler()
    scaler = StandardScaler()
    # TODO: Fit and transform features X into X_scaled
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled,X):
    """
    [Phase 3] K-Means Model Induction
    - Goal: Instantiate KMeans clustering model (e.g., setting n_clusters=3).
    - Fit model to scaled data and assign cluster labels back to dataset.
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    # TODO: Instantiate KMeans(n_clusters=3, random_state=42)
    kmeans = KMeans(n_clusters=3,random_state=42, n_init=10)
    # TODO: Fit model and generate cluster labels (.fit_predict(X_scaled))
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clustered = X.copy()
    df_clustered["cluster"] = cluster_labels
    return kmeans, cluster_labels, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_original):
    """
    [Phase 4] Cluster Evaluation & Business Profiling
    - Goal: Compute Silhouette Score to measure cluster separation and cohesion.
    - Group original data by assigned cluster labels to profile customer segments.
    """
    print("\n[PHASE 4] Cluster Evaluation & Profiling")
    print("-" * 65)
    # TODO: Compute silhouette_score(X_scaled, labels)
    sil_score = silhouette_score(X_scaled,labels)
    # TODO: Append cluster labels to original dataframe and compute group means to profile segments
    cluster_profile = df_original.groupby("cluster").mean(numeric_only=True)
    print("The Cluster Profile is: \n", cluster_profile)
    return sil_score, cluster_profile, df_original


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Unsupervised Model Persistence with Joblib
    - Goal: Save the trained KMeans model and fitted scaler to disk using joblib.
    - Test loading artifacts to predict the cluster segment of a new customer.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    model_dir = Path("./model")
    model_dir.mkdir(exist_ok=True, parents=True)
    # TODO: Save model, scaler, and feature list using joblib.dump()
    joblib.dump(model, model_dir / "customer_segments.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(feature_names, model_dir / "features.joblib")
    # TODO: Load artifacts using joblib.load() and assign a new customer profile to a cluster
    
    loded_model = joblib.load(model_dir / "customer_segments.joblib")
    loaded_scaler = joblib.load(model_dir / "scaler.joblib")
    loaded_feature = joblib.load(model_dir / "features.joblib")
    
    print("Model : \n", loded_model)
    print("Scaler : \n", loaded_scaler)
    print("Feature : \n", loaded_feature)
    return loded_model, loaded_scaler, loaded_feature


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_clustering_pipeline():
    """
    Facade function coordinating the entire unsupervised segmentation pipeline.
    """
    print("=" * 65)
    print("STARTING UNSUPERVISED CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)
    
    # Executing phases chronologically
    X = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, X)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_clustering_pipeline()