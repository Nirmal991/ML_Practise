import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500],
    'bedrooms': [2, 3, 3, 4],
    'age': [5, 4, 3, 2],
    'price': [30, 45, 60, 80]
})

X = df[["area", "bedrooms", "age"]]

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

print(model.coef_)
print(model.intercept_)