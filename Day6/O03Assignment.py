# Assignment : Predict Sales Revenue from amount spent on digital ads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. Dataset Generation
np.random.seed(101)
n_campaigns = 40

ad_spend_k = np.random.uniform(5, 50, size=n_campaigns)  # Ad spend in ₹10,000s
revenue_k = 12 + 3.4 * ad_spend_k + np.random.normal(0, 8, size=n_campaigns)  # Revenue in ₹10,000s

df_marketing = pd.DataFrame({
    "ad_spend_inr_k": ad_spend_k.round(2),
    "revenue_inr_k": revenue_k.round(2)
})
print(df_marketing.head())
print("=" * 60)
# [28.24,30.68,6.28,12.72,35.84]
X = df_marketing[["ad_spend_inr_k"]]
y = df_marketing["revenue_inr_k"].values
print("Value for y is: ", y)
Y = df_marketing["revenue_inr_k"]

model = LinearRegression()
model.fit(X,y)
print(f"Scikit-Learn Slope (Beta 1):     ₹{model.coef_[0]:.2f}")
print(f"Scikit-Learn Intercept (Beta 0): ₹{model.intercept_:.2f}")

print(df_marketing.head())
print("=" * 60)

new_ad = pd.DataFrame({"ad_spend_inr_k": [28.24,30.68,6.28,12.72,35.84]})
predicted_revenue = model.predict(new_ad)
# print(predicted_revenue)

for add, revenue in zip(df_marketing["ad_spend_inr_k"], predicted_revenue):
    print(f"Predicted Revenue for {add} is {revenue}")




