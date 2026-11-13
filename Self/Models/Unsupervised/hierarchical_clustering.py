import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    "Annual_Income": [15, 16, 17, 60, 62, 65, 90, 92, 95],
    "Spending_Score": [20, 22, 25, 75, 78, 80, 30, 32, 35]
}

df = pd.DataFrame(data)

# print(df)
scale = StandardScaler()
X_scaled = scale.fit_transform(df)
# print(X_scaled)

Z = linkage(X_scaled,method='ward')

plt.figure(figsize=(10,6))

dendrogram(Z,labels=df.index.tolist())

plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Customer Index")
# plt.ylabel("Distance")

# plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage="ward")
clusters = model.fit_predict(X_scaled)
df["Cluster"] = clusters
print(df)

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Hierarchical Clustering")

plt.show()