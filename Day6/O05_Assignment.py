import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#1 Synthesize dataset Generation:
np.random.seed(102)
n_tech_samples = 100

years_experience = np.random.uniform(1,12,size=n_tech_samples)
projects_completed = np.random.poisson(lam=5, size=n_tech_samples)
# It generates random whole numbers representing how many times an event occurs within a fixed interval.
code_review_score = np.random.uniform(3.5, 5.0, size=n_tech_samples)

# Target: Salary = Base (40k) + 10k*(exp) + 3k*(projects) + 15k*(review_score) + Noise
annual_salary = (
    40000 
    + 10000 * years_experience 
    + 3000 * projects_completed 
    + 15000 * code_review_score 
    + np.random.normal(0, 4000, size=n_tech_samples)
)

df_tech = pd.DataFrame({
    "experience_years": years_experience.round(1),
    "projects_completed": projects_completed,
    "code_review_score": code_review_score.round(2),
    "annual_salary": annual_salary.round(2)
})

# Extract features:
X_tech = df_tech[["experience_years", "projects_completed", "code_review_score"]]
y_tech = df_tech["annual_salary"]

# 3. Model Building & Fitting
model = LinearRegression()
model.fit(X_tech, y_tech)

#4. Coefficient & Intercept Interpretation
print("--- Tech Salary Multi-Feature Model Coefficients ---")
print(rf"Intercept (Base Salary): ₹{model.intercept_:.2f}")
coef_summary = pd.DataFrame({
    "Feature": X_tech.columns,
    "Partial Coefficient (₹)": model.coef_
})
print(coef_summary.to_string(index=False))

# 5. Interference / Predections on New Engineer
new_engineers = pd.DataFrame({
    "experience_years": [2.5, 8.0],
    "projects_completed": [3, 12],
    "code_review_score": [4.1, 4.9]
})

predictions = model.predict(new_engineers)
new_engineers["Salary_Predictions"] = predictions
# print(new_engineers)
print("\n--- Predictions for New Candidates ---")
# for idx, pred in enumerate(predictions):
#     cand = new_engineers.iloc[idx]
#     print(rf"Candidate {idx+1} ({cand['experience_years']} yrs exp, {cand['projects_completed']} projects, {cand['code_review_score']} rating) -> Estimated Salary: ₹{pred:,.2f}")
    
# compare actual and predicted

y_tech_pred = model.predict(X_tech)
comparison_df = pd.DataFrame({
    "Actual": y_tech,
    "Predicted": y_tech_pred,
    "Difference": y_tech - y_tech_pred
})
print("\n--- Actual vs Predicted Salaries ---")
print(comparison_df.head())

# visualize Comparison
plt.figure(figsize=(10, 6))
plt.scatter(y_tech, y_tech_pred, color='blue', alpha=0.6)
plt.plot([y_tech.min(), y_tech.max()], [y_tech.min(), y_tech.max()], 'r--', lw=2)
plt.xlabel("Actual Salaries")
plt.ylabel("Predicted Salaries")
plt.title("Actual vs Predicted Salaries")
plt.show()