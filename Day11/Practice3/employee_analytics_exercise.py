"""
Title: Employee Productivity & Analytics Lifecycle 
Description: Fill in the TODO sections across the 5 lifecycle phases.

"""

# Step 0: Import necessary libraries
# TODO: Import pandas, numpy, scikit-learn LinearRegression, and metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def phase_1_discovery_and_setup():
    """
    [Phase 1] Discovery & Data Setup (5 Marks)
    - Goal: Load 'employee_productivity.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./employee_productivity.csv")
    
    # TODO: Print first 5 rows and verify shape
    print(df.head())
    print("The shape of the dataset is: \n", df.shape)
    # TODO: Check for missing values
    print("Missing Values are: \n")
    print(df.isna().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures 
    - Goal: Compute measures of central tendency and dispersion 
      for 'Weekly_Tasks_Completed'.
    - Check for potential outliers using IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics ")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, and IQR for Weekly_Tasks_Completed
    summary = df.describe().T
    summary["median"] = df.median(numeric_only=True)
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
    - Goal: Examine relationships between Training Hours, Experience, 
      and Tasks Completed.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association ")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    num_cols = df.select_dtypes(include=[np.number]).columns.values
    corr_matrix = df[num_cols].corr().abs()
    print(corr_matrix)
    # TODO: Highlight the correlation between Training_Hours and Weekly_Tasks_Completed
    correlation = corr_matrix.loc["Training_Hours", "Weekly_Tasks_Completed"]

    print("\nCorrelation between Training Hours and Weekly Tasks Completed:")
    print(correlation)
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling 
    - Goal: Predict 'Weekly_Tasks_Completed' based on 'Training_Hours'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling ")
    print("-" * 65)
    # TODO: Set X = df[['Training_Hours']] and y = df['Weekly_Tasks_Completed']
    X = df[['Training_Hours']]
    y = df['Weekly_Tasks_Completed']
    
    # TODO: Fit Linear Regression model and output slope & intercept
    model = LinearRegression()
    model.fit(X,y)
    print("Coefficient of the model is: ", model.coef_)
    print("Intercept of the model is: ", model.intercept_)
    return model, X, y, df


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendation (5 Marks)
    - Goal: Evaluate model using R-squared and MAE.
    - Provide operational HR recommendations based on data.
    """
    print("\n[PHASE 5] Evaluation & HR Recommendations (5 Marks)")
    print("-" * 65)
    y_pred = model.predict(X)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "r2-score": r2_score(y, y_pred)
    }
    print(metrics)
    # TODO: Print prescriptive action item for employee training investments
    correlation = df["Training_Hours"].corr(
        df["Weekly_Tasks_Completed"]
    )

    if correlation > 0:
        print("\nHR Recommendation:")
        print("Invest in employee training programs because training hours")
        print("show a positive relationship with weekly task completion.")

    elif correlation < 0:
        print("\nHR Recommendation:")
        print("Review the current training strategy because training hours")
        print("show a negative relationship with weekly task completion.")

    else:
        print("\nHR Recommendation:")
        print("Training hours show no linear relationship with task completion.")
        print("Investigate other factors affecting employee productivity.")


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING EMPLOYEE ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y, df = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()