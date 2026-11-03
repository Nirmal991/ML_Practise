"""
Title: E-Commerce Customer Analytics Lifecycle 
Description: Fill in the TODO sections across the 5 lifecycle phases using 
             'ecommerce_customer_data.csv'.

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
    - Goal: Load 'ecommerce_customer_data.csv' and inspect structure.
    - Check for null values and print summary info.
    """
    print("\n[PHASE 1] Discovery & Data Setup")
    print("-" * 65)
    # TODO: Load CSV file using pandas
    df = pd.read_csv("./ecommerce_customer_data.csv")
    # TODO: Print first 5 rows and verify dataset shape
    print(df.head())
    print("The shape of the dataset is: \n", df.shape)
    # TODO: Check for any missing values
    print(df.isnull().sum())
    return df


def phase_2_descriptive_statistics(df):
    """
    [Phase 2] Descriptive Statistics & Data Measures 
    - Goal: Compute measures of central tendency and dispersion 
      for 'Annual_Spend_USD'.
    - Check for potential outliers using the IQR method.
    """
    print("\n[PHASE 2] Descriptive Statistics")
    print("-" * 65)
    # TODO: Calculate Mean, Median, Std Dev, Variance, and IQR for Annual_Spend_USD
    summary = df.describe().T
    summary["median"] = df.median(numeric_only=True)
    summary["variance"] = df.var(numeric_only=True)
    summary["IQR"] = summary["75%"] - summary["25%"]
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
    - Goal: Examine relationships between Browsing Minutes, Items in Cart, and Annual Spend.
    - Compute Pearson correlation matrix.
    """
    print("\n[PHASE 3] Bivariate Association")
    print("-" * 65)
    # TODO: Select numeric columns and compute Pearson correlation matrix
    num_cols = df.select_dtypes(include=[np.number]).columns
    # print("NUM_cols: ", num_cols)
    corr_matrix = df[num_cols].corr().abs()
    # TODO: Highlight correlations involving Annual_Spend_USD
    for col in num_cols:
        correlation = corr_matrix.loc["Annual_Spend_USD", col]
        print("\nCorrelation between Annual_Spend_USD and other columns are: ")
        print(correlation)
    return df


def phase_4_predictive_modeling(df):
    """
    [Phase 4] Predictive Analytics & Regression Modeling 
    - Goal: Predict 'Annual_Spend_USD' based on 'Browsing_Minutes'.
    - Train a Simple Linear Regression model.
    """
    print("\n[PHASE 4] Predictive Modeling")
    print("-" * 65)
    # TODO: Set X = df[['Browsing_Minutes']] and y = df['Annual_Spend_USD']
    X = df[['Browsing_Minutes']]
    y = df['Annual_Spend_USD']
    # TODO: Fit Linear Regression model and output slope & intercept
    model = LinearRegression()
    model.fit(X,y)
    print("Coefficient of the model is: ", model.coef_)
    print("Intercept of the model is: ", model.intercept_)
    return model, X, y, df


def phase_5_evaluation_and_reporting(model, X, y, df):
    """
    [Phase 5] Evaluation & Prescriptive Recommendations 
    - Goal: Evaluate model performance using R-squared and MAE.
    - Provide prescriptive business recommendations based on findings.
    """
    print("\n[PHASE 5] Evaluation & Business Action")
    print("-" * 65)
    y_pred = model.predict(X)
    # TODO: Calculate R2 score and Mean Absolute Error (MAE)
    metrics = {
                "MAE": mean_absolute_error(y, y_pred),
                "r2-score": r2_score(y, y_pred)
            }
    print(metrics)
    # TODO: Print prescriptive marketing action items
    correlations = df["Browsing_Minutes"].corr(df["Annual_Spend_USD"])
    print("Correlations are: ", correlations)
    
    if correlations > 0.5:
        print("Higher Browsing minutes is positively related to Annual_Spend_USD")
        print("Recommend do More Borowsing")
    elif correlations < 0.5:
        print("Higher Browsing minutes is negatively related to Annual_Spend_USD")
        print("More Recommend do More Borowsing")
    else: 
        print("No linear relationship found between Browsing Minutes and Annual_Spend_USD.")


def run_analytics_lifecycle():
    """
    Facade Orchestrator function.
    """
    print("=" * 65)
    print("STARTING E-COMMERCE ANALYTICS LIFECYCLE ")
    print("=" * 65)
    
    # Executing phases chronologically
    df = phase_1_discovery_and_setup()
    df = phase_2_descriptive_statistics(df)
    df = phase_3_bivariate_association(df)
    model, X, y, df = phase_4_predictive_modeling(df)
    phase_5_evaluation_and_reporting(model, X, y, df)


if __name__ == "__main__":
    run_analytics_lifecycle()