
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer
)


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
        self.scaled_cols = []

    def _load_config(self) -> dict:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return config.get("numerical_transformation", {})

    def fit_transform(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        exclude_cols: list = None
    ) -> pd.DataFrame:

        df_clean = df.copy()

        exclude_cols = exclude_cols or []

        if is_train:

            numeric_cols = df_clean.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            numeric_cols = [
                col for col in numeric_cols
                if col not in exclude_cols
                and df_clean[col].nunique() > 2
            ]

        else:
            numeric_cols = self.transformed_cols

        # ----------------------------------
        # 1. Power Transformation
        # ----------------------------------

        power_cfg = self.config.get("power_transform", {})

        if power_cfg.get("enabled", False):

            method = power_cfg.get("method", "yeo-johnson")

            if is_train:

                self.power_transformer = PowerTransformer(
                    method=method,
                    standardize=False
                )

                self.transformed_cols = numeric_cols.copy()

                if self.transformed_cols:

                    df_clean[self.transformed_cols] = (
                        self.power_transformer.fit_transform(
                            df_clean[self.transformed_cols]
                        )
                    )

                logger.info(
                    f"Power transformation applied: {self.transformed_cols}"
                )

            else:

                if self.power_transformer is None:
                    raise ValueError(
                        "PowerTransformer has not been fitted."
                    )

                if self.transformed_cols:

                    df_clean[self.transformed_cols] = (
                        self.power_transformer.transform(
                            df_clean[self.transformed_cols]
                        )
                    )

        # ----------------------------------
        # 2. Feature Scaling
        # ----------------------------------

        scaling_cfg = self.config.get("scaling", {})

        if scaling_cfg.get("enabled", False):

            method = scaling_cfg.get("method", "standard")

            if is_train:

                if method == "standard":
                    self.scaler = StandardScaler()

                elif method == "robust":
                    self.scaler = RobustScaler()

                else:
                    raise ValueError(
                        f"Unsupported scaling method: {method}"
                    )

                self.scaled_cols = numeric_cols.copy()

                if self.scaled_cols:

                    df_clean[self.scaled_cols] = (
                        self.scaler.fit_transform(
                            df_clean[self.scaled_cols]
                        )
                    )

                logger.info(
                    f"Feature scaling applied: {self.scaled_cols}"
                )

            else:

                if self.scaler is None:
                    raise ValueError(
                        "Scaler has not been fitted."
                    )

                if self.scaled_cols:

                    df_clean[self.scaled_cols] = (
                        self.scaler.transform(
                            df_clean[self.scaled_cols]
                        )
                    )

        return df_clean
