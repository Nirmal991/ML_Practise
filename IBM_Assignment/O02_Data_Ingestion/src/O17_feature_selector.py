import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    chi2,
    SelectKBest
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

logger = logging.getLogger("FeatureSelector")


class FeatureSelector:

    def __init__(self, config_path: Path):

        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.variance_selector = None
        self.variance_input_cols = []
        self.variance_selected_cols = []
        self.variance_fitted = False

        self.correlation_dropped_cols = []
        self.correlation_fitted = False

        self.statistical_selector = None
        self.statistical_input_cols = []
        self.selected_features = []
        self.statistical_fitted = False

    def _load_config(self) -> dict:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        return config.get("feature_selection", {})

    # --------------------------------------------------
    # Helper: Identify numerical feature columns
    # --------------------------------------------------

    def _get_numeric_features(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> list:

        return [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col != target_col
        ]

    # --------------------------------------------------
    # STEP 1: Variance Threshold
    # --------------------------------------------------

    def apply_variance_threshold(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        target_col: str = "Churn"
    ) -> pd.DataFrame:

        var_cfg = self.config.get("variance_threshold", {})

        if not var_cfg.get("enabled", True):
            return df.copy()

        threshold = var_cfg.get("threshold", 0.01)

        df_clean = df.copy()

        if is_train:

            numeric_cols = self._get_numeric_features(
                df_clean,
                target_col
            )

            self.variance_input_cols = numeric_cols.copy()

            if not numeric_cols:
                self.variance_selected_cols = []
                self.variance_fitted = True
                return df_clean

            self.variance_selector = VarianceThreshold(
                threshold=threshold
            )

            self.variance_selector.fit(
                df_clean[numeric_cols]
            )

            self.variance_selected_cols = [
                col
                for col, keep in zip(
                    numeric_cols,
                    self.variance_selector.get_support()
                )
                if keep
            ]

            dropped_cols = [
                col
                for col in numeric_cols
                if col not in self.variance_selected_cols
            ]

            self.variance_fitted = True

            logger.info(
                f"VarianceThreshold dropped: {dropped_cols}"
            )

        else:

            if not self.variance_fitted:
                raise ValueError(
                    "VarianceThreshold not fitted. Run training first."
                )

            missing_cols = [
                col
                for col in self.variance_input_cols
                if col not in df_clean.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"Missing variance input columns: {missing_cols}"
                )

            dropped_cols = [
                col
                for col in self.variance_input_cols
                if col not in self.variance_selected_cols
            ]

        df_clean.drop(
            columns=dropped_cols,
            inplace=True
        )

        return df_clean

    # --------------------------------------------------
    # STEP 2: Correlation Filtering
    # --------------------------------------------------

    def apply_correlation_filter(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        target_col: str = "Churn"
    ) -> pd.DataFrame:

        corr_cfg = self.config.get("correlation_filter", {})

        if not corr_cfg.get("enabled", True):
            return df.copy()

        threshold = corr_cfg.get("threshold", 0.85)

        df_clean = df.copy()

        if is_train:

            numeric_cols = self._get_numeric_features(
                df_clean,
                target_col
            )

            num_df = df_clean[numeric_cols]

            if num_df.shape[1] <= 1:
                self.correlation_dropped_cols = []
                self.correlation_fitted = True
                return df_clean

            corr_matrix = num_df.corr().abs()

            upper_tri = corr_matrix.where(
                np.triu(
                    np.ones(corr_matrix.shape),
                    k=1
                ).astype(bool)
            )

            self.correlation_dropped_cols = [
                column
                for column in upper_tri.columns
                if (upper_tri[column] > threshold).any()
            ]

            self.correlation_fitted = True

            logger.info(
                f"Correlation filter dropped: "
                f"{self.correlation_dropped_cols}"
            )

        else:

            if not self.correlation_fitted:
                raise ValueError(
                    "Correlation filter not fitted. Run training first."
                )

        missing_cols = [
            col
            for col in self.correlation_dropped_cols
            if col not in df_clean.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing correlation-filter columns: {missing_cols}"
            )

        df_clean.drop(
            columns=self.correlation_dropped_cols,
            inplace=True
        )

        return df_clean

    # --------------------------------------------------
    # STEP 3: Statistical Feature Selection
    # --------------------------------------------------

    def apply_statistical_scoring(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_train: bool = True
    ) -> pd.DataFrame:

        stat_cfg = self.config.get("statistical_scoring", {})

        method = stat_cfg.get("method", "mutual_info")
        top_k = stat_cfg.get("top_k", 15)

        df_clean = df.copy()
        
        print("\n========== MISSING VALUE CHECK ==========")

        

        if is_train:

            if target_col not in df_clean.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found."
                )

            feature_cols = self._get_numeric_features(
                df_clean,
                target_col
            )

            if not feature_cols:
                raise ValueError(
                    "No numerical features available for selection."
                )

            X = df_clean[feature_cols]
            y = df_clean[target_col]

            if y.isna().any():
                raise ValueError(
                    "Target contains missing values."
                )

            self.statistical_input_cols = feature_cols.copy()

            if len(feature_cols) <= top_k:

                self.selected_features = feature_cols.copy()
                self.statistical_selector = None

                logger.info(
                    "Feature count <= top_k. Keeping all features."
                )

            else:

                if method == "mutual_info":

                    score_func = lambda X, y: mutual_info_classif(
                        X,
                        y,
                        random_state=42
                    )

                elif method == "chi2":

                    if (X < 0).any().any():
                        raise ValueError(
                            "Chi-square requires non-negative features."
                        )

                    score_func = chi2

                else:
                    raise ValueError(
                        f"Unsupported statistical method: {method}"
                    )

                self.statistical_selector = SelectKBest(
                    score_func=score_func,
                    k=top_k
                )

                self.statistical_selector.fit(X, y)

                self.selected_features = [
                    col
                    for col, keep in zip(
                        feature_cols,
                        self.statistical_selector.get_support()
                    )
                    if keep
                ]

                logger.info(
                    f"Selected top {top_k} features: "
                    f"{self.selected_features}"
                )

            self.statistical_fitted = True

        else:

            if not self.statistical_fitted:
                raise ValueError(
                    "Statistical selector not fitted. "
                    "Run training first."
                )

            missing_cols = [
                col
                for col in self.statistical_input_cols
                if col not in df_clean.columns
            ]

            if missing_cols:
                raise ValueError(
                    f"Missing statistical input columns: {missing_cols}"
                )

        # Preserve target and non-numerical columns
        non_feature_cols = [
            col
            for col in df_clean.columns
            if col not in self.statistical_input_cols
        ]

        df_clean = df_clean[
            non_feature_cols + self.selected_features
        ]

        return df_clean