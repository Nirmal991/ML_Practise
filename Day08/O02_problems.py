import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

def generated_agriculture_data(n_samples: int = 200, seed: int = 2026) -> pd.DataFrame:
    np.random.seed(seed)
    fertilizer = np.random.uniform(20, 250, size=n_samples)
    # Yield = 2.0 + 0.08*Fertilizer - 0.00028*(Fertilizer^2) + Noise
    true_yield = 2.0 + 0.08 * fertilizer - 0.00028 * (fertilizer**2)
    observed_yield = true_yield + np.random.normal(0, 0.8, size=n_samples)
    df = pd.DataFrame({
                "fertilizer_kg_ha": fertilizer.round(1),
                "crop_yield_tons": observed_yield.round(2),
            } )
    print("Generated agricultural data:")
    print(df.head())
    print("=" * 60)
    return df

def train_polynomial_yield_model(df: pd.DataFrame,
                                feature_col:str = "fertilizer_kg_ha",
                                target_col: str = "crop_yield_tons",
                                degree: int = 2,
                                test_size: float = 0.2,
                                ) -> Tuple[Pipeline,
                                           pd.DataFrame,
                                           pd.DataFrame,
                                           pd.Series,
                                           pd.Series,
                                           Dict[str, float]]:
    if feature_col not in df.columns or target_col not in df.columns:
        raise ValueError("Specified feature or target column not found in the DataFrame.")
    
    X = df[[feature_col]]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=2026)
    pipeline = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred)
    }
    print("Polinomial model training completed")
    return pipeline, X_train,X_test,y_test, y_train, metrics

def predict_farm_yields(
    pipeline: Pipeline, new_farm_data: pd.DataFrame
) -> pd.DataFrame:
    predictions = pipeline.predict(new_farm_data)
    results_df = new_farm_data.copy()
    results_df["predicted_yield_tons"] = predictions.round(2)
    return results_df

def main():
    #1. Ingest Data
    df_crop = generated_agriculture_data(n_samples=250)
    
    #2. Train Polynomial Model Pipeline (Degree 2)
    pipeline, X_train, X_test, y_train, y_test, metrics = train_polynomial_yield_model(
        df_crop, degree=2)
    
    # 3. Model Performance Evaluation                                                                    df_crop, degree=2)
    print("\n--- Model Performance Metrics (Test Set) ---")
    print(f"Polynomial RMSE : {metrics['RMSE']:.4f} Tons/ha")
    print(f"R² Score        : {metrics['R2']:.4f}")
    
    #for predictions:
    new_farm_plots = pd.DataFrame( {
            "fertilizer_kg_ha": [50.0, 140.0, 220.0,115.0,70.4]
        } )
    predicted_yields = predict_farm_yields(pipeline, new_farm_plots)
    print("\n--- Predicted Yields for New Farm Plots ---")
    print(predicted_yields.to_string(index = False))
    print("=" * 60)
    

if __name__ == "__main__":
    main()