import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.DataFrame({
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "salary": [
        20000, 25000, 30000, 40000, 45000,
        50000, 60000, 65000, 75000, 80000
    ]
})

X = df[['experience']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print(
    "RMSE:",
    np.sqrt(mean_squared_error(y_test, y_pred))
)

print("R2:", r2_score(y_test, y_pred))

# One useful advantage of Decision Trees is that they can estimate feature importance.
# Feature importance tells us how much each feature contributes to reducing impurity across the tree.
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)