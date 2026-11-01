import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("OutlierDetector")

class OutlierDetector:
    def __init__(self, config_path:Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileExistsError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("outlier_detection", {})
        
    def detect_and_treat_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Treats outliers using Z-Score bounds (clipping values exceeding threshold)."""
        z_cfg = self.config.get("z_score", {})
        threshold = z_cfg.get("threshold", 3.0)
        cols_to_check  = self.config.get("numerical_columns_to_check", [])
        
        df_clean = df.copy()
        logger.info(f"Running Z-Score outlier treatment (threshold = {threshold})...")
        
        for col in cols_to_check:
            if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            
            if std_val == 0 or pd.isna(std_val):
                continue
            
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            outliers_count = (z_scores > threshold).sum()
            
            if outliers_count > 0:
                upper_bound = mean_val + (threshold * std_val)
                lower_bound = mean_val - (threshold * std_val)
                df_clean[col] = df_clean[col].clip(lower_bound=lower_bound, upper_bound=upper_bound)
                logging.info(f"Clipped {outliers_count} Z-Score outliers in '{col}' to bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
                
        return df_clean
    
    def detect_and_treat_iqr(self, df: pd.DataFrame)->pd.DataFrame:
        iqr_cfg = self.config.get("iqr", {})
        multiplier = iqr_cfg.get("multiplier", 1.5)
        cols_to_check = self.config.get("numerical_columns_to_check", [])
        
        df_clean = df.copy()
        logger.info(f"Running IQR outlier treatment (multiplier = {multiplier})...")
        
        for col in cols_to_check:
            if col not in df_clean.columns or pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            print(f"Processing column: {col}")
            print(f"Column dtype: {df_clean[col].dtype}")
            
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - (multiplier*IQR)
            upper_bound = Q3 + (multiplier*IQR)
            
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col]>upper_bound)).sum()
            
            if outliers_count > 0:
                df_clean[col] = df_clean[col].clip(lower_bound=lower_bound, upper_bound=upper_bound)
                logger.info(f"Clipped {outliers_count} IQR outliers in '{col}' to fence [{lower_bound:.2f}, {upper_bound:.2f}].")
                
        return df_clean
    
    def detect_multivariant_isolation_forest(self, df: pd.DataFrame)->pd.DataFrame:
        iso_cfg = self.config.get("isolation_forest", {})
        contamination = iso_cfg.get("contamination", 0.02)
        random_state = iso_cfg.get("random_state", 42)
        cols_to_check = [c for c in self.config.get("numerical_columns_to_check", []) if c in df.columns]
        
        df_clean = df.copy()
        if not cols_to_check:
            return df_clean
    
        logger.info(f"Running Isolation Forest multivariate anomaly detection (contamination={contamination})...")
        
        X = df_clean[cols_to_check].select_dtypes(include=np.number)
        if X.empty():
            return df_clean
        
        model = IsolationForest(contamination=contamination, random_state=random_state)
        preds = model.fit_predict(X)
        
        # IsolationForest returns -1 for anomalies and 1 for normal points
        df_clean["is_multivariate_anomaly"] = (preds == -1).astype(int)
        anomaly_count = df_clean["is_multivariate_anomaly"].sum()
        
        logger.info(f"Isolation Forest flagged {anomaly_count:,} multivariate anomalous records.")
        return df_clean
        