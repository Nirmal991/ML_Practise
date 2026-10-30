# How to decide the best eps in the dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# 1. Real-world dataset: Customer Annual Income (k$) vs. Spending Score (1-100)
data = {
    'Annual_Income': [
        15,        16,        17,        18,        19,        20,
        25,        26,        28,        30,        50,        52,
        54,        55,        58,        60,        80,        82,
        85,        88,        90,        95,        110,       115,
        120,       125,
    ],
    'Spending_Score': [
        39,        81,        6,        77,        40,        76,
        72,        55,        60,        42,        50,        55,        
        48,        52,        54,        49,        15,        10,        
        12,        14,        18,        16,        80,        85,        
        79,        83,
    ],
}

df = pd.DataFrame(data)
print(df.head())

'''
Golden rule before applying DBSCAN: always standardize your data.
DBSCAN is sensitive to the scale of the data, so standardization is crucial.
DBSCAN relies on distance to calculate eps.
If the data is not standardized, features with larger scales will dominate the distance calculation, leading to suboptimal clustering results.
'''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Annual_Income', 'Spending_Score']])

# 2. Apply DBSCAN
'''
DBSCAN parameters:
- eps: The maximum distance between two samples for them to be considered as in the same neighborhood. :eps = epsilon: the radius of the neighborhood around a point.
- min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
'''
dbscan = DBSCAN(eps=0.5, min_samples=3)
clusters = dbscan.fit_predict(X_scaled)
df['Cluster'] = clusters
print(df)

# min_pts = 5

# neighbours = NearestNeighbors(n_neighbors=min_pts)
# neighbours_fit = neighbours.fit(X_scaled)

# distances, indices = neighbours_fit.kneighbors(X_scaled)

# k_distances = np.sort(distances[:, -1])
# print("Nearest Distance:", k_distances)

# plt.figure(figsize=(8, 5))

# plt.plot(k_distances)

# plt.xlabel("Data Points (Sorted)")
# plt.ylabel("K-Distance")
# plt.title("K-Distance Graph for Finding Optimal EPS")

# plt.grid(True)
# plt.show()


# plot result
# 4. Plot the results   
plt.figure(figsize=(8, 6))
plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster'],
    cmap='tab10',
    edgecolor='k',
    s=70,
)
plt.title('DBSCAN Customer Segmentation (Real Data)')
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
