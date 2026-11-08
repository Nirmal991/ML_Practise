import logging
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from sklearn.impute import KNNImputer

logging.basicConfig(
    level=logging.INFO,
    level="%(asctime)s [%(levelname)s] [%(name)s]: %(messsage)s"
)
logger = logging.getLogger("MissingnessHandler")
'''
missingness_rules:
  knn_imputer:
    n_neighbors: 5
    weights: "uniform"

  indicator_flagging_cols:
    - "age"
    - "cabin"
    - "embarked"

  mnar_sentinel_value: -999.0
'''

class MissingnessHandler:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("missingness_rules", {})
        
    def analyze_missigness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating missingness audit report...")
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        
        report_df = {
            "missing_count": missing_counts,
            "missing_percentage": missing_percentage
        }
        report_df = report_df[report_df["missing_count"] > 0].sort_index(by="missing_percentage", ascending=False)
        
        if report_df.empty:
            logger.info("No missing value Exists....")
        else:
            logger.warning(f"Columns with missing data is: \n{report_df}")
        
        return report_df
    
    def treat_missigness(self, df: pd.DataFrame):
        pass
        