import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [40, 45, 50, 55, 65, 70, 75, 80, 90, 95],
    "result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[["study_hours", "attendance"]]

y = df["result"]

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

model = LogisticRegression(
    penalty="l2",  #	Regularization technique
    C=1.0,   # 	Inverse of regularization strength
    solver="lbfgs",   #	Optimization algorithm
    max_iter=1000 # 	Maximum optimization iterations
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(y_pred)

y_probability = model.predict_proba(X_test_scaled)

print(y_probability)

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

print("ROC-AUC:",
      roc_auc_score(
          y_test,
          model.predict_proba(X_test_scaled)[:, 1]
      ))