import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

'''domain_features:
  ratios_to_build:
    - numerator: "TotalCharges"
      denominator: "tenure"
      new_feature_name: "charge_per_tenure_month"
    - numerator: "MonthlyCharges"
      denominator: "tenure"
      new_feature_name: "monthly_to_tenure_ratio"'''


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
        if not self.config_path.exists():
          raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
          return yaml.safe_load(f).get("domain_features", {})
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
      df_clean = df.copy()
      
      ratios = self.config.get("ratios_to_build", [])
      for ratio in ratios:
        num_col = ratio.get("numerator")
        denom_col = ratio.get("denominator")
        new_col = ratio.get("new_feature_name")
        
        if num_col not in df_clean.columns or denom_col not in df_clean.columns:
          continue
        
        df_clean[num_col] = pd.to_numeric(
            df_clean[num_col], errors="coerce"
        )

        df_clean[denom_col] = pd.to_numeric(
            df_clean[denom_col], errors="coerce"
        )
        
        denom_series = df_clean[denom_col].replace(0,np.nan)
        df_clean[new_col] = df_clean[num_col] / denom_series
        df_clean[new_col] = df_clean[new_col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
      return df_clean