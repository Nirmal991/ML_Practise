import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
print(X)
y = np.array([2, 5, 10, 17, 26, 37])
print(y)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# print(X_poly)

model = LinearRegression()
model.fit(X_poly,y)

y_pred = model.predict(X_poly)
print("Predictions:", y_pred)

mse = mean_squared_error(y, y_pred)

r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

# plt.scatter(X, y, label="Actual Data")

# plt.plot(X, y_pred, color="red", label="Polynomial Regression")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Polynomial Regression")
# plt.legend()
# plt.show()

for degree in range(1, 6):

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    model.fit(X_poly, y)

    predictions = model.predict(X_poly)

    mse = mean_squared_error(y, predictions)

    print(f"Degree: {degree}, MSE: {mse}")
