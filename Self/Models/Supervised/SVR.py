import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = {
    "House_Size": [
        500, 700, 900, 1100, 1300,
        1500, 1700, 1900, 2100, 2300
    ],

    "Price": [
        20, 25, 32, 38, 45,
        55, 65, 75, 85, 100
    ]
}

df = pd.DataFrame(data)

print(df)

X = df[["House_Size"]]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)

X_test_scaled = scaler_X.transform(X_test)

svr_model = SVR(
    kernel="rbf", # 	Specifies the kernel function.
    C=100,#  Controls the penalty for errors outside epsilon.
    epsilon=0.1, # Defines the acceptable error margin.
    gamma="scale"# Controls the influence of individual observations for nonlinear kernels.
)

svr_model.fit(X_train_scaled, y_train)

y_pred = svr_model.predict(X_test_scaled)

print("Predicted Values:")
print(y_pred)

print("Actual Values:")
print(y_test.values)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2)

