import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

# 1. Create a DataFrame for the sample data points
data = {
    'Sample_ID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    'Feature_A': [1, 1, 1, 4, 4, 4],
    'Feature_B': [2, 4, 0, 2, 4, 0]
}

df = pd.DataFrame(data)
print(df)

# 2 Seprate what is require for clustering and labeling:
features = df[['Feature_A', 'Feature_B']]
sample_labels = df['Sample_ID'].values

#3 Compute the hierarchial/agglomerative clustering using ward`s method
linked = linkage(features, 'ward')

org_dist = pdist(features)

c_coeff, cophetic_dist = cophenet(linked, org_dist)
print(f"\nCophenetic Correlation Coefficient: {c_coeff:.4f}")

# 4. Plot the dendrogram with DataFrame labels
plt.figure(figsize=(8, 5))
dendrogram(linked,
           labels=sample_labels,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)

plt.title("Hierarchical Clustering Dendrogram from DataFrame")
plt.xlabel("Sample ID")
plt.ylabel("Euclidean Distance")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()