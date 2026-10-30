import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

X = pd.DataFrame({"study_hours": [1,2,4,6,7]})
y = np.array([50,60,70,80,60])

print(pd.DataFrame({"Hours": X["study_hours"], "score": y}))
print("="*60)


# Compare Degrees 1, 2,  4
degrees = [1, 2, 4]
# model = make_pipeline(PolynomialFeatures(degree =1 ), LinearRegression())
# model = make_pipeline(PolynomialFeatures(degree =2 ), LinearRegression())
# model = make_pipeline(PolynomialFeatures(degree =4 ), LinearRegression())

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree),LinearRegression)
    model.fit(X,y)
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    print(f"Degree {degree} RMSE: {rmse:.2f}")
    
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"Degree {degree} RMSE: {rmse:.2f}")