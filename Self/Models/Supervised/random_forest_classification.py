import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 3, 5, 6, 8, 9],

    "attendance": [40, 45, 50, 55, 60, 65, 70, 80,
                   85, 95, 42, 52, 62, 68, 82, 90],

    "previous_score": [30, 35, 40, 45, 55, 60, 65, 75,
                       80, 95, 32, 42, 58, 63, 78, 88],

    "result": [0, 0, 0, 0, 1, 1, 1, 1,
               1, 1, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

y_prob = model.predict_proba(X_test)

print(y_prob)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

print("\nROC-AUC:")

print(
    roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)


# plt.figure(figsize=(8, 5))

# plt.bar(
#     feature_importance["Feature"],
#     feature_importance["Importance"]
# )

# plt.xlabel("Features")
# plt.ylabel("Importance")

# plt.title("Random Forest Feature Importance")

# plt.xticks(rotation=45)

# plt.show()
