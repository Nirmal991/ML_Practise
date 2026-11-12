import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.features_names
)

y = iris.target

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="euclidean"
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(y_test, y_pred)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

error_rate = []

for k in range(1, 21):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    error = np.mean(predictions != y_test)

    error_rate.append(error)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 21),
    error_rate,
    marker="o"
)

plt.xlabel("K Value")

plt.ylabel("Error Rate")

plt.title("KNN: Error Rate vs K")

plt.xticks(range(1, 21))

plt.grid()

plt.show()  