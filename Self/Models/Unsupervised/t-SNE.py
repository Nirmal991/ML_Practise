import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

iris = load_iris()

X = iris.data
y = iris.target

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    max_iter=1000,
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

print(X_tsne.shape)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="viridis"
)

plt.title("t-SNE Visualization of Iris Dataset")

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.colorbar(label="Flower Class")

plt.show()