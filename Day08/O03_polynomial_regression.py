import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

# Feature X: Fertilizer amount applied (kg per acre)
X = pd.DataFrame({"fertilizer_kg": [1, 2, 3, 4, 5, 6, 7, 8]})

# Target y: Resulting crop yield (bushels) - notice how it peaks at 5 kg then drops!
y = [15, 28, 38, 45, 48, 46, 40, 30]

best_degree = None
lowest_rmse = float("inf")
for deg in [1,2,4]:
    model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE for degree {deg}: {rmse}")
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_degree = deg
        
print(f"Best degree: {best_degree} with RMSE: {lowest_rmse}") # this gives the lowest RMSE among the tested degrees
print("=" * 60)

model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model.fit(X,y)

new_farms = pd.DataFrame({"fertilizer_kg": [2.5, 4.5, 6.5]})
predicted_yield = model.predict(new_farms)

df = X.copy()
df["crop_yield"] = y
df = pd.concat([df, new_farms.assign(crop_yield=predicted_yield)], ignore_index=True)   
print(df)

import matplotlib.pyplot as plt
import numpy as np
x_smooth = pd.DataFrame({"fertilizer_kg": np.linspace(df["fertilizer_kg"].min(), df["fertilizer_kg"].max(), 300)})
y_smooth = model.predict(x_smooth)
# 2. Your original plot elements + the smooth line
plt.scatter(df["fertilizer_kg"], df["crop_yield"], color="blue", label="Actual")
plt.scatter(new_farms["fertilizer_kg"], predicted_yield, color="red", label="Predicted")
# Plot the smooth bell line
plt.plot(x_smooth["fertilizer_kg"], y_smooth, color="green", linewidth=2, label="Fitted Curve (Bell)")
plt.xlabel("Fertilizer (kg per acre)")
plt.ylabel("Crop Yield (bushels)")
plt.title("Polynomial Regression Predictions")
plt.legend()
plt.show()