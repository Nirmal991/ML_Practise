import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("EDAProfiler")

class EDAProfiler:
    def __init__(self):
        logger.info("Initialized EDAProfiler.")
        
    def generate_univariate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Generating univariate statistical profiling summary...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warning("No numerical columns found for univariate summary.")
            return pd.DataFrame()
        
        summary = numeric_df.describe().T
        summary["skewness"] = numeric_df.skew()
        summary["missing_counts"] = numeric_df.isna().sum()
        summary["missing_percentage"] = (numeric_df.isnull().sum() / len(numeric_df)) * 100
        
        logger.info("Univariate summary generated successfully.")
        return summary
    
    def compute_correlation_matrix(self, df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        logger.info(f"Computing bivariate correlation matrix using method: {method}...")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        corr_matrix = numeric_df.corr(method=method)
        logger.info("Correlation matrix computed successfully.")
        return corr_matrix
    
    def analyze_target_interactions(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in DataFrame.")
            return pd.DataFrame()
        
        logger.info(f"Analyzing feature interactions against target: '{target_col}'...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop([target_col], errors="ignore")
        
        interactions = []   
        for col in numeric_cols:
            grouped = df.groupby(target_col)[col].mean()
            interactions.append({
                "feature": col,
                "mean_by_target": grouped.to_dict()
            })
            
        logger.info("Target interaction analysis complete.")
        return pd.DataFrame(interactions)
    
if __name__ == "__main__":
    # Local test demonstration
    sample_df = pd.DataFrame({
        "tenure_months": [12, 24, 6, 48, 3],
        "monthly_charges": [29.95, 75.0, 19.0, 105.0, 45.0],
        "total_charges": [359.4, 1800.0, 114.0, 5040.0, 135.0],
        "churn_status": [0, 1, 0, 0, 1]
    })
    
    
    
    print("\n[RESULT] Univariate Summary Preview:")
    print(summary)
    print("\n[RESULT] Correlation Matrix Preview:")
    print(corr)