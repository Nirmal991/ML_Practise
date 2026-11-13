import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

X,y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

df = pd.DataFrame(X,columns=["Annual_Income", "Spending_Score"])

# print(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.3, min_samples=5, metric="euclidean")

clusters = model.fit_predict(X_scaled)

df["Clusters"] = clusters
# print(df.head())

# plt.figure(figsize=(8,6))

# plt.scatter(
#     X_scaled[:,0],
#     X_scaled[:,1],
#     c=clusters,
#     cmap='viridis',
#     s=50
# )

# plt.title("DBSCAN Clustering")
# plt.xlabel("Annual Income (Scaled)")
# plt.ylabel("Spending Score (Scaled)")

# plt.show()

# Finding Optimal Epsilon Using K-Distance Graph
min_samples = 5
neighbours = NearestNeighbors(n_neighbors=min_samples)
neighbours.fit(X_scaled)
distances, indices = neighbours.kneighbors(X_scaled)
k_distances = np.sort(distances[:,-1])

# plt.figure(figsize=(8, 6))

# plt.plot(k_distances)

# plt.xlabel("Data Points Sorted by Distance")
# plt.ylabel("5th Nearest Neighbor Distance")

# plt.title("K-Distance Graph")

# plt.show()

n_clusters = len(set(clusters) - {-1})

if n_clusters >= 2:

    mask = clusters != -1

    score = silhouette_score(
        X_scaled[mask],
        clusters[mask]
    )

    print("Silhouette Score:", score)

else:
    print("Silhouette Score cannot be calculated.")
    
n_clusters = len(set(clusters) - {-1})

print("Number of Clusters:", n_clusters)

n_noise = np.sum(clusters == -1)

print("Noise Points:", n_noise)
