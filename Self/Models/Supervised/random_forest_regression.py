import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = {
    "House_Size": [1000, 1200, 1500, 1800, 2000,
                   2200, 2500, 2800, 3000, 3500],

    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],

    "House_Age": [10, 8, 7, 5, 4, 5, 3, 2, 1, 1],

    "Price": [30, 35, 45, 55, 65,
              70, 85, 95, 110, 130]
}

df = pd.DataFrame(data)

# print(df)

X = df.drop(columns="Price")
print(X)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_depth=None,min_samples_split=2, min_samples_leaf=1,max_features=1, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Actual Values:")
print(y_test.values)

print("Predicted Values:")
print(y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2)

importance = rf_model.feature_importances_

feature_imortance = pd.DataFrame({
    "Feature": X.columns,
    "importance": importance
})

print(feature_imortance)

feature_imortance = feature_imortance.sort_values(
    by="importance",
    ascending=False
)
