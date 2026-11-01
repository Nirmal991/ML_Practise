import logging
import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

logger = logging.getLogger("CardinalityReducer")


class CardinalityDimensionalityReducer:

    def __init__(self, config_path: Path):

        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.frequent_categories = {}

        self.pca_transformer = None
        self.pca_input_cols = []
        self.pca_output_cols = []

    def _load_config(self) -> dict:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return config.get("cardinality_reduction", {})

    # --------------------------------------------------
    # 1. Cardinality Reduction
    # --------------------------------------------------

    def group_rare_categories(
        self,
        df: pd.DataFrame,
        categorical_cols: list,
        is_train: bool = True
    ) -> pd.DataFrame:

        threshold = self.config.get("rare_label_threshold", 0.02)

        df_clean = df.copy()

        existing_cols = [
            col for col in categorical_cols
            if col in df_clean.columns
        ]

        if not existing_cols:
            return df_clean

        if is_train:

            self.frequent_categories = {}

            for col in existing_cols:

                frequencies = df_clean[col].value_counts(
                    normalize=True
                )

                frequent = frequencies[
                    frequencies >= threshold
                ].index.tolist()

                self.frequent_categories[col] = frequent

                df_clean[col] = df_clean[col].apply(
                    lambda x: x if pd.isna(x) or x in frequent else "Other"
                )

            logger.info(
                f"Rare category grouping completed for: {existing_cols}"
            )

        else:

            if not self.frequent_categories:
                raise ValueError(
                    "Frequent categories not set. "
                    "Run group_rare_categories with is_train=True first."
                )

            for col in existing_cols:

                frequent = self.frequent_categories.get(col, [])

                df_clean[col] = df_clean[col].apply(
                    lambda x: x if pd.isna(x) or x in frequent else "Other"
                )

        return df_clean

    # --------------------------------------------------
    # 2. PCA Dimensionality Reduction
    # --------------------------------------------------

    def apply_pca_reduction(
        self,
        df: pd.DataFrame,
        variance_ratio: float = 0.95,
        is_train: bool = True,
        exclude_cols: list = None
    ) -> pd.DataFrame:

        df_clean = df.copy()

        exclude_cols = exclude_cols or []

        if is_train:

            numeric_cols = df_clean.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            self.pca_input_cols = [
                col for col in numeric_cols
                if col not in exclude_cols
            ]

            if len(self.pca_input_cols) <= 2:
                return df_clean

            logger.info(
                f"Fitting PCA to retain {variance_ratio * 100}% variance."
            )

            self.pca_transformer = PCA(
                n_components=variance_ratio,
                random_state=42
            )

            pca_array = self.pca_transformer.fit_transform(
                df_clean[self.pca_input_cols]
            )

            self.pca_output_cols = [
                f"pc_{i+1}"
                for i in range(pca_array.shape[1])
            ]

            pca_df = pd.DataFrame(
                pca_array,
                columns=self.pca_output_cols,
                index=df_clean.index
            )

            df_clean = pd.concat(
                [
                    df_clean.drop(columns=self.pca_input_cols),
                    pca_df
                ],
                axis=1
            )

            logger.info(
                f"PCA reduced numerical features from "
                f"{len(self.pca_input_cols)} to "
                f"{len(self.pca_output_cols)} components."
            )

        else:

            if self.pca_transformer is None:
                raise ValueError(
                    "PCA transformer not fitted. "
                    "Run on training data first."
                )

            missing_cols = [
                col for col in self.pca_input_cols
                if col not in df_clean.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"Missing PCA input columns in test data: {missing_cols}"
                )

            logger.info(
                "Transforming test dataset using fitted PCA transformer."
            )

            pca_array = self.pca_transformer.transform(
                df_clean[self.pca_input_cols]
            )

            pca_df = pd.DataFrame(
                pca_array,
                columns=self.pca_output_cols,
                index=df_clean.index
            )

            df_clean = pd.concat(
                [
                    df_clean.drop(columns=self.pca_input_cols),
                    pca_df
                ],
                axis=1
            )

        return df_clean

