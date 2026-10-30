import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DomainFeatureBuilder")

class DomainFeatureBuilder:
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Loads domain feature engineering rules from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("domain_features", {})
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        ratios = self.config.get("ratios_to_build", [])
        print("Ratios: ", ratios)
        for ratio in ratios:
            num_col = ratio.get("numerator")
            denom_col = ratio.get("denominator")
            new_col = ratio.get("new_feature_name")
            
            if num_col not in df_clean.columns and denom_col not in df_clean.columns:
                continue
            
            denom_series = df_clean[denom_col].replace(0, np.nan)
            df_clean[new_col] = df_clean[num_col] / denom_series
            df_clean[new_col] = df_clean[new_col].replace([np.inf,-np.inf], np.nan).fillna(0)
            
        return df_clean
    
if __name__ == "__main__":
    config_file = Path(__file__).parent.parent / "config/O01_feature_rules.yaml"
    sample_df = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002", "CUST-003"],
        "tenure_months": [12, 0, 24],  # 0 tenure tests division-by-zero protection
        "monthly_charges": [29.95, 75.0, 45.0],
        "total_charges": [359.4, 0.0, 1080.0]
    })
    
    if config_file.exists():
        builder = DomainFeatureBuilder(config_path=config_file)
        df_featured = builder.build_features(sample_df)
        print("\n[RESULT] Domain Engineered DataFrame Preview:")
        print(df_featured)
    else: 
        logger.error(f"Configuration file not found: {config_file}")