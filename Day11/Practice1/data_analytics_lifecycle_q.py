"""
Title: End-to-End Data Analytics Lifecycle
Description: A modular Python script demonstrating business discovery, 
             data preparation, exploration, statistics, and predictive modeling.
"""

# ==========================================
# IMPORTS (Libraries for Data, Stats, & Viz)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


# ==========================================
# SUB-STEP FUNCTIONS (The Lifecycle Phases)
# ==========================================

def phase_1_discovery_and_setup():
    """
    Module 1: Foundations, Business Understanding & Setup
    - Define business objective (e.g., Coffee Shop Revenue Analysis).
    - Load the 25-record real-world CSV dataset.
    - Initial inspection (head, info, missing values).
  
    """
    print("Executing Phase 1: Discovery & Data Setup...")
    # TODO: Load dataset using pandas
    df = pd.read_csv("./coffee_shop_data.csv")
    # TODO: Perform basic inspection (shape, data types, missing values)
    print("The head of the data set is: ")
    print(df.head())
    print()
    print("The shape of the dataset is: ")
    print(df.shape)
    print()
    print("The data types are: ")
    print(df.dtypes)
    print()
    print("The null values are: ")
    print(df.isna().sum())
    return df


def phase_2_descriptive_statistics(df: pd.DataFrame):
    """
    Module 2: Descriptive Statistics & Data Measures
    - Calculate Central Tendency (Mean, Median, Mode).
    - Calculate Dispersion (Standard Deviation, Variance, IQR, Range).
    - Check Distribution Shape & Outliers.
  
    """
    print("Executing Phase 2: Descriptive Statistics & Measures...")
    # TODO: Compute summary statistics (mean, median, std, IQR)
    # summary = pd.DataFrame(df.describe().T)
    summary = df.describe().T
    # summary["median"] = summary["50%"]
    summary["median"] = df.median(numeric_only=True)
    summary["mode"] = df.mode(numeric_only=True).iloc[0]
    summary["variance"] = df.var(numeric_only=True)
    summary["IQR"] = summary["75%"] - summary["25%"]
    summary["range"] = summary["max"] - summary["min"]
    print(summary)
    # TODO: Identify and handle any outliers
    num_cols = df.select_dtypes(include=[np.number]).columns.values
    # print(num_cols)
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5*iqr
        # print(upper_bound)
        lower_bound = q1 - 1.5*iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound) 
    return df


def phase_3_bivariate_association(df: pd.DataFrame):
    """
    Module 4: Bivariate & Multivariate Association
    - Analyze relationships between variables (e.g., Marketing Spend vs. Revenue).
    - Compute Covariance and Pearson Correlation Matrix.
    - Generate exploratory visualizations (Scatter plots, Correlation heatmap).
  
    """
    print("Executing Phase 3: Bivariate Association & Visualization...")
    # TODO: Compute correlation matrix
    df_clean = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.values
    corr_matrix = df[num_cols].corr().abs()
    print("Correltion matrix is: ", corr_matrix)
    # TODO: Plot correlation heatmap and scatter plots
    plt.figure(figsize=(10,6))
    sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    target = "Daily_Revenue_USD"
    features = [
        "Temperature_C",
        "Foot_Traffic",
        "Marketing_Spend_USD"
    ]
    
    for col in features:
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=df_clean,x=col,y=target)
        plt.title(f"{col} vs {target}")
        plt.xlabel(col)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
    return df_clean

def phase_4_predictive_modeling(df: pd.DataFrame):
    """
    Module 6 & 8: Predictive Analytics & Regression Modeling
    - Set up a simple predictive model (e.g., Simple Linear Regression).
    - Train the model on independent variables to predict revenue.
    - Establish a baseline evaluation.
    """
    
    print("Executing Phase 4: Predictive Modeling...")
    print("Available columns:", df.columns.tolist())
    # TODO: Define features (X) and target (y)
    X = df[["Temperature_C", "Foot_Traffic", "Marketing_Spend_USD"]]
    y = df["Daily_Revenue_USD"]
    # TODO: Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    # TODO: Extract model coefficients and intercept
    print("The intercept of the model is: ", model.intercept_)
    print("The coefficient of the model is: ", model.coef_)

    y_pred = model.predict(X)
    print(y_pred)
    return df,y, y_pred


def phase_5_evaluation_and_reporting(df: pd.DataFrame, y: pd.Series, y_pred: np.ndarray):
    """
    Module 6 & 7: Model Evaluation, Generalization & Business Decision
    - Evaluate model performance (e.g., R-squared, Mean Absolute Error).
    - Discuss generalization and overfitting risks on small data.
    - Translate technical findings into prescriptive business recommendations.
    [Allocation: 5 Marks]
    """
    print("Executing Phase 5: Evaluation & Business Action...")
    # TODO: Calculate evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y,y_pred),
        "RMSE": root_mean_squared_error(y, y_pred),
        "r2-score": r2_score(y, y_pred)
    }
    print(metrics)
    # TODO: Print final business recommendations based on findings
    correlations = df.corr(numeric_only=True)["Daily_Revenue_USD"]
    # print("Correlations: ", correlations)
    # print(correlations["Foot_Traffic"])
    print("\nBUSINESS RECOMMENDATIONS")
    print("="*50)
    
    #Food Traffic
    if correlations["Foot_Traffic"] > 0.5:
        print("1. Foot Traffic has a strong positive relationship with revenue.")
        print("Recommendation: Invest in strategies to increase customer visits.")

    elif correlations["Foot_Traffic"] < -0.5:
        print("1. Foot Traffic has a strong negative relationship with revenue.")
        print("Recommendation: Investigate why increased traffic is associated with lower revenue.")

    else:
        print("1. Foot Traffic has a weak relationship with revenue.")
        print("Recommendation: Investigate other factors influencing customer spending.")
    pass


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_analytics_lifecycle():
    """
    Facade function that coordinates the entire data analytics 
    lifecycle in chronological order.
    """
    print("=" * 60)
    print("STARTING COMPLETE DATA ANALYTICS LIFECYCLE (Out of 40 Marks)")
    print("=" * 60)
    
    df = phase_1_discovery_and_setup()
    print("-" * 60)
    
    df = phase_2_descriptive_statistics(df)
    print("-" * 60)
    
    df = phase_3_bivariate_association(df)
    print("-" * 60)
    
    df, y, y_pred = phase_4_predictive_modeling(df)
    print("-" * 60)
    
    phase_5_evaluation_and_reporting(df, y, y_pred)
    
    print("=" * 60)
    print("ANALYTICS LIFECYCLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()