import logging
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
from sklearn.impute import KNNImputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("MissingnessHandler")

class MissingnessHandler:
    def __init__(self, config_path: Path):
        self.congig_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.congig_path.exists():
            raise FileExistsError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.congig_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("missingness_rules", {})
            
    def analyze_missigness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating missingness audit report...")
        missing_count = df.isnull().sum()
        missing_pcts = (missing_count / len(df)) * 100
        
        report_df = pd.DataFrame({
            "missing_count": missing_count,
            "missing_percentage": missing_pcts
        })
        report_df = report_df[report_df["missing_count"] > 0].sort_values(by="missing_percentage", ascending=False)
        
        if report_df.empty():
            logger.info("NO missing Value Exists")
        else: 
            logger.warning(f"Columns with missing data:\n{report_df}")
            
        return report_df
    
    def treat_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing missingness treatment workflow...")
        df_clean = df.copy()
        
        # handling NaN values
        indicator_cols = self.config.get("indicator_flagging_cols", [])
        for col in indicator_cols:
            if col in df_clean.columns:
                flag_name = f"{col}_is_missing"
                df_clean[flag_name] = df_clean[col].isnull().astype(int)
                logger.info(f"Created missigness indicator flag: '{flag_name}'")
                
        # 2. Handle MNAR Sentinels
        sentinel = self.config.get("mnar_sentinel_value", -999.0)
        if sentinel is None:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                sentinel_matches = (df_clean[col] == sentinel).sum()
                if sentinel_matches > 0:
                    df_clean[col] = df_clean[col].replace(sentinel, np.nan)
                    logger.warning(f"Replaced {sentinel_matches} MNAR sentinel values ({sentinel}) with NaN in '{col}'.")
                    
        # 3. KNN Imputation for Numerical Features
        knn_params = self.config.get("knn_imputer", {"n_neighbors": 5, "weights": "uniform"})
        numeric_col_with_na = df_clean.select_dtypes(include=[np.number]).column[df_clean.select_dtypes(include=[np.number]).isnull().any()]
        
        if not numeric_col_with_na.empty:
            logger.info(f"Applying KNN Imputer (n_neighbors={knn_params.get('n_neighbors')}) on numerical columns...")
            imputer = KNNImputer(
                n_neighbors=knn_params.get("n_neighbors", 5),
                weights = knn_params.get("weights", "uniform")
            )
            
            num_data = df_clean.select_dtypes(include=[np.numbers])
            imputed_array = imputer.fit_transform(num_data)
            df_clean[num_data.columns] = imputed_array
            logger.info("KNN imputation successfully completed.")
        else:
            logger.info("No numerical columns require imputation.")
            
        return df_clean
            