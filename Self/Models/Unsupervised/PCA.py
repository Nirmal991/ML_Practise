import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# print(X.head())

# print("Original Shape:", X.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Reduced Shape:", X_pca.shape)

print("Explained Variance Ratio:")

# print(pca.explained_variance_ratio_)

pca_full = PCA()

X_full = pca_full.fit_transform(X_scaled)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

# print(cumulative_variance)

# plt.figure(figsize=(8, 5))

# plt.plot(
#     range(1, len(cumulative_variance) + 1),
#     cumulative_variance,
#     marker="o"
# )

# plt.xlabel("Number of Principal Components")

# plt.ylabel("Cumulative Explained Variance")

# plt.title("PCA Cumulative Explained Variance")

# plt.grid()

# plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=40
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA - Breast Cancer Dataset")

plt.show()