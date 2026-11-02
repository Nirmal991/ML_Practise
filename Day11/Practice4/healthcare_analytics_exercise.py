"""
Title: Healthcare Patient Analytics Lifecycle 
Description: Fill in the TODO sections across the 5 lifecycle phases using 
             'patient_health_data.csv'.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, LinearRegression from sklearn.linear_model, and evaluation metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup 
    - Goal: Load 'patient_health_data.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./patient_health_data.csv")
    # TODO: Print first 5 rows and verify dataset shape
    print(df.head())
    print("The shape of the dataset is: \n", df.shape)
    # TODO: Check for any missing values
    print("Missing Values are: \n")
    print(df.isna().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures 
    - Goal: Compute measures of central tendency and dispersion 
      for 'Systolic_BP'.
    - Check for potential outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, Variance, and IQR for Systolic_BP
    summary = df.describe().T
    summary["median"] = summary["50%"]
    summary["IQR"] = summary["75%"] - summary["25%"]
    print(summary)
    # TODO: Identify any values falling outside 1.5 * IQR bounds
    num_cols = df.select_dtypes(include=[np.number]).columns.values
        
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
            
        IQR = q3 - q1
            
        upper_bound = q3 + 1.5*IQR
        print(upper_bound)
        lower_bound = q1 - 1.5*IQR
            
        outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
            
        print("Name of column: ", col)
        print("Number of Outliers:", len(outliers))
    return df


def phase_3_bivariate_association(df):
    """
    [Phase 3] Bivariate Association & Correlation 
    - Goal: Examine relationships between BMI, Daily Steps, and Systolic BP.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    num_cols = df.select_dtypes(include=[np.number]).columns.values
    corr_matrix = df[num_cols].corr().abs()
    print(corr_matrix)
    # TODO: Highlight correlations involving Systolic_BP
    for col in num_cols:
        correlation = corr_matrix.loc[f"Systolic_BP", col]
    print("\nCorrelation between Systolic_BP and other columns are: ")
    print(correlation)
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling 
    - Goal: Predict 'Systolic_BP' based on 'BMI'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)
    # TODO: Set X = df[['BMI']] and y = df['Systolic_BP']
    X = df[['BMI']]
    y = df['Systolic_BP']
    # TODO: Fit Linear Regression model and output slope & intercept
    model = LinearRegression()
    model.fit(X,y)
    print("Coefficient of the model is: ", model.coef_)
    print("Intercept of the model is: ", model.intercept_)
    return model, X, y, df


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendations (5 Marks)
    - Goal: Evaluate model performance using R-squared and MAE.
    - Provide prescriptive clinical recommendations based on findings.
    """
    print("\n[PHASE 5] Evaluation & Clinical Action (5 Marks)")
    print("-" * 65)
    y_pred = model.predict(X)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "r2-score": r2_score(y, y_pred)
        }
    print(metrics)
    # TODO: Print prescriptive clinical action items
    correlation = df["Systolic_BP"].corr(df["Age"])

    if correlation > 0:
        print("Higher age is positively associated with systolic BP.")
        print("Recommend regular blood pressure monitoring.")

    elif correlation < 0:
        print("Age is negatively associated with systolic BP.")
        print("Further clinical assessment is recommended.")

    else:
        print("No linear relationship found between Age and Systolic BP.")
        print("Investigate other clinical risk factors.")
    pass


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING HEALTHCARE ANALYTICS LIFECYCLE ")
    print("=" * 65)
    
    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y, df = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()