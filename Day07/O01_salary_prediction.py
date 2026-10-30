from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

def generate_synthetic_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    years_experience = np.random.uniform(1, 15, size=n_samples)
    certifications = np.random.poisson(lam=2, size=n_samples)
    performance_rating = np.random.uniform(3.0, 5.0, size=n_samples)
    hours_per_week = np.random.normal(40, 4, size=n_samples)
      # Underlying formula with noise:
    # Base 45k + 9.5k*(Exp) + 3.5k*(Certs) + 12k*(Perf) + 200*(Hours) + Gaussian Noise
    salary = (
        45000
        + 9500 * years_experience
        + 3500 * certifications
        + 12000 * performance_rating
        + 200 * hours_per_week
        + np.random.normal(0, 6000, size=n_samples)
    )
    df = pd.DataFrame({
        "years_experience": years_experience,
        "certifications": certifications,
        "performance_rating": performance_rating,
        "hours_per_week": hours_per_week,
        "salary": salary
    })
    return df

def predict_new_candidate_salary(model:LinearRegression, candidate_data:pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(candidate_data)
    result_df = candidate_data.copy()
    result_df["predicted_salary"] = predictions
    return result_df

def train_salary_model(
    df:pd.DataFrame, target_col: str = "salary", test_size: float = 0.2
) -> Tuple[LinearRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str,float],]:
    #STEP 1:
    if target_col not in df.columns:
        raise ValueError(f"Terget Column '{target_col}' not found in DataFrame")
    
    #Step 2:
    df = df.copy()
    
    #STEP 3:
    X = df.drop(columns=[target_col]) # prdictors
    y = df["salary"] # target columns
    
    #STEP 4:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    
    #STEP 5:
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    #STEP 6: Make prediction on test set
    y_pred = model.predict(X_test)
    
    #STEP 7:
    n = len(y_test)
    p = X_train.shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1-r2) * (n-1)/ (n-p-1)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2,
        "Adjusted_R2":adj_r2,
    }
    return model, X_train, X_test, y_train, y_test, metrics


def main():
    print("=" * 60)
    print("Salary Prediction")
    df = generate_synthetic_data()
    print(df.head())
    print("=" * 60)
    model, X_train, X_test, y_train, y_test, metrics = train_salary_model(df)
    
    # print("Evaluation Metrics: ")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value}")
    df_metrics = pd.DataFrame([metrics])
    print("Evaluation Metrics: ")
    print(df_metrics.T)
    
    print("="*60)
    print("Base intercept: ", model.intercept_)
    coef_df = pd.DataFrame({
        "Features": X_train.columns,
        "Weight_learned": model.coef_
    }).sort_values(by="Weight_learned", ascending=False)
    print("Model Coefficient: ")
    print(coef_df.to_string(index=False))
    print("="*60)
    
    # Predict Salary for a new Candidate
    new_candidate_data = pd.DataFrame([{
        "years_experience": 3.5,
        "certifications": 1,
        "performance_rating": 3.8,
        "hours_per_week": 42,
    },
    {
        "years_experience": 9.0,
        "certifications": 4,
        "performance_rating": 4.7,
        "hours_per_week": 45.0,
    }])
    
    inference_result = predict_new_candidate_salary(model, new_candidate_data)
    print("Inference Results: ")
    print(inference_result.to_string(index=False))
    print("="*60)
if __name__ == "__main__":
    main()