import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("NumericalSanitizer")


class NumericalSanitizer:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads mathematical sanitization parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("mathematical_sanitization", {})
        
    def safe_divide(self, numerator: pd.Series, denominator: pd.Series, default_val = 0.0) -> pd.Series:
        denom_clean = denominator.replace(0, np.nan)
        result = numerator / denom_clean
        result = result.fillna(default_val)
        return result
    
    def sanitize_infinities_and_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        
        mask_inf = self.config.get("mask_inf_with_nan", True)
        df_clean = df.copy()
        
        logger.info("Scanning for IEEE 754 infinity and NaN values in numerical columns...")
        numeric_col = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_col:
            inf_count = np.isinf(df_clean[col]).sum()
            if inf_count > 0:
                if mask_inf:
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    logger.warning(f"Replaced {inf_count:,} infinite values with NaN in column '{col}'.")
                else:
                    default_val = self.config.get("zero_division_default", 0.0)
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], default_val)
                    logger.warning(f"Replaced {inf_count:,} infinite values with default ({default_val}) in column '{col}'.")
                    
                    
        return df_clean