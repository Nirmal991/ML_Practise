import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

df = pd.DataFrame({
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [
        20000, 25000, 30000, 40000, 45000,
        50000, 60000, 65000, 75000, 80000
    ]
})

X = df[["experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted Salary:", y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


plt.scatter(X, y, label="Actual Data")
plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression Line"
)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.legend()
plt.show()

new_data = pd.DataFrame({
    "experience": [11, 12, 13]
})

predictions = model.predict(new_data)
print("Predicted Salaries for New Data:", predictions)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)


lasso = Lasso(alpha=1.0)

lasso.fit(X_train, y_train)