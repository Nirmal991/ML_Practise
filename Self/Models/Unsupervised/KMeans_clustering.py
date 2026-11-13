import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'Player': [
        'Player1', 'Player2', 'Player3', 'Player4', 'Player5',
        'Player6', 'Player7', 'Player8', 'Player9', 'Player10',
        'Player11', 'Player12', 'Player13', 'Player14', 'Player15'
    ],

    'Runs': [
        5000, 4500, 3800, 2000, 1800,
        1500, 1200, 900, 700, 600,
        300, 250, 150, 100, 50
    ],

    'Wickets': [
        5, 10, 8, 80, 70,
        90, 100, 120, 150, 160,
        200, 180, 220, 250, 270
    ],

    'Batting_Average': [
        45, 42, 38, 35, 32,
        30, 28, 25, 20, 18,
        12, 10, 8, 6, 5
    ],

    'Strike_Rate': [
        90, 85, 88, 80, 78,
        75, 72, 70, 65, 60,
        55, 50, 45, 40, 35
    ],

    'Bowling_Average': [
        50, 45, 48, 30, 32,
        28, 27, 25, 22, 20,
        18, 19, 17, 16, 15
    ]
}

df = pd.DataFrame(data)

# print(df)

X = df[['Runs', 'Wickets', 'Batting_Average','Strike_Rate', 'Bowling_Average']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)

df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_analysis = df.groupby('Cluster')[['Runs', 'Wickets', 'Batting_Average','Strike_Rate', 'Bowling_Average']].mean()

print(cluster_analysis)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Runs',
    y='Wickets',
    hue='Cluster',
    palette='viridis',
    s=150
)

plt.title('K-Means Cricket Player Clustering')
plt.xlabel('Total Runs')
plt.ylabel('Total Wickets')

plt.show()

wcss = []

for k in range(1, 10):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(range(1, 10), wcss, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')

plt.show()