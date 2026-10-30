import logging
from pathlib import Path
import joblib
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def build_pipelines(clustering_cfg:dict) -> Pipeline:
    return Pipeline([
        ("scaler",StandardScaler()),
        ("hierarchical", AgglomerativeClustering(
            n_clusters=clustering_cfg.get("n_clusters",2),
            metric=clustering_cfg.get("metric", "euclidean"),
            linkage=clustering_cfg.get("linkage","ward")
        ))
    ])
    
def compute_cophenetic_correlation(X_scaled, linkage_model:str) -> float:
    pairwise_dicts = pdist(X_scaled, metric="euclidean")
    Z = linkage(X_scaled, method=linkage_model)
    coph_corr, _ = cophenet(Z, pairwise_dicts)
    return coph_corr

def profile_clusters(
    df: pd.DataFrame, labels: pd.Series, features: list
) -> pd.DataFrame:
    df_profile = df[features].copy()
    df_profile["cluster_id"] = labels
    profile = df_profile.groupby("cluster_id").mean().round(2)
    profile["sample_count"] = df_profile.groupby("cluster_id").size()
    return profile

def main():
    config = load_config()
    data_config = config.get("data", {})
    clustering_cfg = config.get("clustering", {})
    
    input_file = Path(data_config["input_path"])
    if not input_file.exists():
        raise FileNotFoundError(f"Input data file not found at {input_file}")
    df = pd.read_csv(input_file)
    feature_cols = data_config.get("features", [])
    
    X = df[feature_cols]
    print("Original Data:")
    print(df.head())
    print("=" * 60)
    # selected features
    print("Selected Features:")
    print(X.head())
    print("=" * 60)
    
    pipeline = build_pipelines(clustering_cfg)
    labels = pipeline.fit_predict(X)
    # labels: contains the cluster number assigned to each data point by the Agglomerative Clustering algorithm.
    print("Cluster Labels:")
    print(labels)
    print("=" * 60)
    
    # calculate the evaluation diagnosis:
    scaled_data = pipeline.named_steps["scaler"].transform(X)
#     pipeline.named_steps["scaler"] accesses the StandardScaler object inside your pipeline.
#    .transform(X) applies standardization to your original features.
#     scaled_data stores the standardized features.
    sil_score = silhouette_score(scaled_data, labels)
    # Purpose: Measures how well-separated and compact your clusters are.
    db_index = davies_bouldin_score(scaled_data, labels)    
    # It checks whether clusters are compact internally and sufficiently separated from each other.
    # Measures how similar each cluster is to its most similar other cluster, considering cluster spread and separation
    coph_corr = compute_cophenetic_correlation(scaled_data, clustering_cfg["linkage"])
    print("Evaluation Diagnostics:")
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Cophenetic Correlation: {coph_corr}")
    print("=" * 60)
    profiles = profile_clusters(df, labels, feature_cols)
    print("Cluster Profiles:")
    print(profiles)
    print("=" * 60)
    outer_path = Path(data_config["output_model_path"])

# Create parent directory if it doesn't exist
    outer_path.parent.mkdir(parents=True, exist_ok=True)

# Save the model
    joblib.dump(pipeline, outer_path)

    print(f"Model saved to {outer_path.resolve()}")
    
    loaded_pipeline = load_joblib_model(str(outer_path))
    loaded_labels = loaded_pipeline.fit_predict(X)
    print("Loaded Model Cluster Labels:")
    print(loaded_labels)
    print("=" * 60)

def load_joblib_model(path:str):
    return joblib.load(path)

if __name__ == "__main__":
    main()

