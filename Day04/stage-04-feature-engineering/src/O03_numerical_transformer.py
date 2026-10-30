import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("NumericalTransformer")

class NumericalTransformer:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.power_transformer = None
        self.scaler = None
        self.transformed_cols = []

    def _load_config(self) -> dict:
        """Loads numerical transformation parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("numerical_transformation", {})

        def fit_transform_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
            df_clean = df.copy()
            return df_clean

if __name__ == "__main__":
    pass