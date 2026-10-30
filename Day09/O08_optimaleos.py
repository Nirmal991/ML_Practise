import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_wine
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
print(df.head())

X_raw =df[['alcohol', 'malic_acid']]

# golden rule: always standardize your data before applying DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

min_samples = 5
neigh = NearestNeighbors(n_neighbors=min_samples)
neigh.fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled) # _ = indices

k_distances = np.sort(distances[:, -1]) # sort the k-distances in ascending order # farthest neighbor distance
x = range(len(k_distances))

knee_locator = KneeLocator(x, k_distances, curve='convex', direction='increasing')
optimal_eps = knee_locator.knee_y

print("Optimal eps:", optimal_eps)

dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)

clusters = dbscan.fit_predict(X_scaled)
df['cluster'] = clusters
print(df.head())
print(df.tail())

#5
plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    X_raw['alcohol'], X_raw['malic_acid'], c=clusters, cmap='tab10', edgecolor='k', s=60
)
plt.title(
    f'DBSCAN on Real Wine Data\n(Auto-detected eps = {optimal_eps:.2f}, min_samples = {min_samples})'
)
plt.xlabel('Alcohol Content')
plt.ylabel('Malic Acid')
plt.colorbar(scatter, label='Cluster ID (-1 = Noise)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()