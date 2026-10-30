import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

# step 1 Create a simple Dataset with 4 natural groups (clusters)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
'''
Parameters for make_blobs:
    - n_samples: Total number of data points to generate.
    - centers: Number of distinct clusters to generate.
    - cluster_std: Standard deviation of the clusters.
    - random_state: Seed for reproducibility.
'''
print("=" * 60)

# 2. Test different numbers of clusters (k from 2 to 6)
k_values = range(2,7)
best_k = None
best_score = float('inf')# Start with infinity so any score is lower
best_lebels = None

for k in k_values:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    print(f"k={k}, DBI={score}")
    if score < best_score:
        best_score = score
        best_k = k
        best_lebels = labels
        
print("=" * 60)
print(f"Best k: {best_k} with DBI: {best_score}")


# 3. Visualize the clusters for the optimal k
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=best_lebels, cmap='viridis', s=50, edgecolor='k')
plt.title(f"Hierarchical Clustering (Optimal k = {best_k})\nDavies-Bouldin Score: {best_score:.4f}")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()