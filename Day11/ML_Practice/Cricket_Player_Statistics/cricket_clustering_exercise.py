"""
Title: Unsupervised Cricket Player Role Segmentation 
Description: A modular Python exercise script for loading cricket statistics, 
             scaling features, training a K-Means clustering model to group 
             players into functional roles (Batsmen, Bowlers, All-Rounders), 
             evaluating clusters, and persisting artifacts using joblib.
"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, StandardScaler from sklearn.preprocessing, 
#       KMeans from sklearn.cluster, silhouette_score from sklearn.metrics, and joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery, Setup & Feature Selection
    - Goal: Load 'cricket_player_stats.csv' and inspect structure.
    - Separate player names/identifiers from numeric statistics.
    - Select numeric features required for role clustering.
    """
    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./cricket_player_stats.csv")
    # TODO: Retain 'player_name' for final profiling
    y = df["player_name"]
    # TODO: Select numerical features (e.g., 'runs_scored', 'batting_average', 'wickets_taken', 'bowling_economy')
    X = df[["runs_scored", "batting_average", "wickets_taken", "bowling_economy"]]
    return X, df


def phase_2_feature_scaling(X):
    """
    [Phase 2] Unsupervised Feature Scaling
    - Goal: Fit StandardScaler on performance metrics to equalize feature weightings.
    """
    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)
    # TODO: Initialize StandardScaler()
    scaler = StandardScaler()
    # TODO: Fit and transform features X into X_scaled
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def phase_3_kmeans_model_induction(X_scaled, df_original):
    """
    [Phase 3] K-Means Model Induction
    - Goal: Train KMeans model setting n_clusters=3 (Batsmen, Bowlers, All-Rounders).
    - Fit model to scaled data and assign cluster labels back to dataset.
    """
    print("\n[PHASE 3] K-Means Model Induction & Labeling")
    print("-" * 65)
    # TODO: Instantiate KMeans(n_clusters=3, random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42)
    # TODO: Fit model and generate cluster labels using .fit_predict(X_scaled)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clustered = df_original.copy()
    df_clustered["cluster"] = cluster_labels
    return kmeans, cluster_labels, df_clustered


def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered):
    """
    [Phase 4] Cluster Evaluation & Role Profiling
    - Goal: Calculate Silhouette Score to measure cluster separation.
    - Calculate average statistics per cluster and translate centroids into cricket roles.
    """
    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)
    # TODO: Compute silhouette_score(X_scaled, labels)
    sil_score = silhouette_score(X_scaled,labels)
    print("\nSilhouette Score:", sil_score)
    # TODO: Group df_clustered by 'cluster_id' and compute mean statistics to interpret roles
    cluster_profile = df_clustered.groupby("cluster").mean(numeric_only=True)
    
    print("\nAverage Statistics Per Cluster:") 
    print(cluster_profile)
    
    for cluster_id, row in cluster_profile.iterrows():
        print(f"\nCluster {cluster_id}:")
        
        runs = row["runs_scored"]
        batting_avg = row["batting_average"]
        wickets = row["wickets_taken"]

        if runs >= 5000 and batting_avg >= 30 and wickets >= 50:
            role = "All-Rounder"

        elif runs >= 5000 and batting_avg >= 30:
            role = "Batsman"

        else:
            role = "Bowler"
        print("Predicted Cricket Role:", role)
    return sil_score, cluster_profile


def phase_5_joblib_model_persistence(model, scaler, feature_names):
    """
    [Phase 5] Unsupervised Model Persistence with Joblib
    - Goal: Save the trained KMeans model and scaler to disk using joblib.
    - Test loading artifacts to assign a new player profile to a role cluster.
    """
    print("\n[PHASE 5] Model Persistence with Joblib")
    print("-" * 65)
    model_dir = Path("./model")
    model_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Save model, scaler, and feature list using joblib.dump()
    joblib.dump(model, model_dir / "cricket_stats.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(feature_names, model_dir / "feature_names.joblib")
    # TODO: Load artifacts using joblib.load() and predict the cluster for a new player profile
    loaded_model = joblib.load(model_dir / "cricket_stats.joblib")
    loaded_scaler = joblib.load(model_dir / "scaler.joblib")
    loaded_features = joblib.load(model_dir / "feature_names.joblib")
    print("\nSaved artifacts loaded successfully!")
    
    print("\nModel:", loaded_model)
    print("Scaler:", loaded_scaler)
    print("Features:", loaded_features)
    return loaded_model, loaded_scaler, loaded_features


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_cricket_clustering_pipeline():
    """
    Facade function coordinating the cricket player segmentation pipeline.
    """
    print("=" * 65)
    print("STARTING CRICKET PLAYER ROLE CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)
    
    # Executing phases chronologically
    X, df_original = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, df_original)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_cricket_clustering_pipeline()