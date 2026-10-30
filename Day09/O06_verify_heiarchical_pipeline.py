import joblib
import pandas as pd
from pathlib import Path

#1. load serialized pipeline artifact
model_path = Path("hierarchical_pipeline.joblib")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path.resolve()}")

pipeline = joblib.load(model_path)
print(f" Successfully loaded pipeline from: {model_path.name}")
print(f"Pipeline Steps: {list(pipeline.named_steps.keys())}\n")

# 2. Inspect fitted components
scaler = pipeline.named_steps["scaler"]
clusterer = pipeline.named_steps["hierarchical"]

# 3. Create out-of-sample test input
test_data = pd.DataFrame([
    {"feature_1": 13.0, "feature_2": 11.0, "feature_3": 110.0, "feature_4": 2.0},  # Similar to Cluster 0
    {"feature_1": 46.0, "feature_2": 52.0, "feature_3": 315.0, "feature_4": 6.0},  # Similar to Cluster 1
    {"feature_1": 84.0, "feature_2": 76.0, "feature_3": 485.0, "feature_4": 9.2},  # Similar to Cluster 2
])

# 4. Transform & Predict
# Standardize features using fitted scaler
X_test_scaled = pipeline.named_steps["scaler"].transform(test_data)


training_features = pipeline.named_steps["scaler"].feature_names_in_
# 
# feature_names_in_: This attribute stores the names of the features that were provided when the scaler was fitted using a Pandas DataFrame.
print("--- Standardized Test Features ---")
print(pd.DataFrame(X_test_scaled, columns=training_features).round(2))


# Extract cluster labels for training dataset stored inside the fitted step
train_labels = clusterer.labels_
print(f"\nTraining Cluster Assignment Counts:\n{pd.Series(train_labels).value_counts().to_dict()}")