import logging
from pathlib import Path
import pandas as pd
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
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("missingness_rules", {})
        
    def analyze_missigness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating missigness audit report...")
        missing_counts = df.isnull().sum()
        missing_pcts = ( missing_counts/len(df)) * 100
        
        report_df = pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pcts": missing_pcts
        })
        report_df = report_df[report_df["missing_count"] > 0].sort_values(by="missing_pcts", ascending=False)
        
        if report_df.empty:
            logger.info("No missing values detected in dataset.")
        else: 
            logger.info(f"Columns with message data:\n{report_df}")
            
        return report_df
    
    def treatment_missigness(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing missigness treatment workflow...")
        df_clean = df.copy()
        
        indicator_cols = self.config.get("indicator_flagging_cols", [])
        for col in indicator_cols:
            if col in df_clean.columns:
                flag_name = f"{col}_is_missing"
                df_clean[flag_name] = df_clean[col].isnull().astype(int)
                logger.info(f"Created missingness indicator flag: '{flag_name}'")
                
        sentinel = self.config.get("mnar_sentinel_value")
        if sentinel is not None:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                sentinel_matches = (df_clean[col] == sentinel).sum()
                if sentinel_matches > 0:
                    df_clean[col] = df_clean[col].replace(sentinel, np.nan)
                    logger.warning(f"Replaced {sentinel_matches} MNAR sentinel values ({sentinel}) with NaN in '{col}'.")
                    
        knn_params = self.config.get("knn_imputer", {"n_neighbors": 5, "weights": "uniform"})
        numeric_col_with_na = df_clean.select_dtypes(include=[np.number]).columns[df_clean.select_dtypes(include=[np.number]).isnull().any()]
        
        if not numeric_col_with_na.empty:
            logger.info(f"Applying KNN Imputer (n_neighbour={knn_params.get('n_neighbors')}) on numerical columns...")
            imputer = KNNImputer(
                n_neighbors=knn_params.get("n_neighbours", 5),
                weights=knn_params.get("weights", "uniform")
            )
            num_data = df_clean.select_dtypes(include=[np.number])
            imputed_array = imputer.fit_transform(num_data)
            df_clean[num_data.columns] = imputed_array
            logger.info("KNN imputation successfully completed.")
        else:
            logger.info("No numerical columns required imputation.")
        return df_clean
    
if __name__ == "__main__":
    # config_file = Path("config/O01_cleaning_rules.yaml")
    config_file = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "O01_cleaning_rules.yaml"
)
    
    sample_df = pd.DataFrame({
        "tenure_months": [12, np.nan, 24, 6],
        "monthly_charges": [29.95, 75.0, -999.0, 45.0], # -999.0 represents MNAR sentinel
        "total_charges": [359.4, np.nan, np.nan, 270.0]
    })
    
    print("Current working directory:", Path.cwd())
    print("Config file path:", config_file.resolve())
    print("Config file exists:", config_file.exists())
    
    if config_file.exists():
        handler = MissingnessHandler(config_path=config_file)
        handler.analyze_missigness_report(sample_df)
        treated_df = handler.treatment_missigness(sample_df)
        print("\n[RESULT] Treated DataFrame Preview:")
        print(treated_df)