import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)

logger = logging.getLogger("OutlierDetector")


class OutlierDetector:

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        return config_data.get("outlier_detection", {})

    def detect_and_treat_zscore(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:

        z_cfg = self.config.get("z_score", {})
        threshold = z_cfg.get("threshold", 3.0)

        cols_to_check = self.config.get(
            "numerical_columns_to_check", []
        )

        df_clean = df.copy()

        logger.info(
            f"Running Z-Score outlier treatment "
            f"(threshold={threshold})..."
        )

        for col in cols_to_check:

            if (
                col not in df_clean.columns
                or not pd.api.types.is_numeric_dtype(df_clean[col])
            ):
                continue

            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()

            if std_val == 0 or pd.isna(std_val):
                continue

            z_scores = np.abs(
                (df_clean[col] - mean_val) / std_val
            )

            outliers_count = (z_scores > threshold).sum()

            if outliers_count > 0:

                upper_bound = mean_val + threshold * std_val
                lower_bound = mean_val - threshold * std_val

                df_clean[col] = df_clean[col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )

                logger.info(
                    f"Clipped {outliers_count} Z-Score outliers "
                    f"in '{col}' to bounds "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}]."
                )

        return df_clean

    def detect_and_treat_iqr(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:

        iqr_cfg = self.config.get("iqr", {})
        multiplier = iqr_cfg.get("multiplier", 1.5)

        cols_to_check = self.config.get(
            "numerical_columns_to_check", []
        )

        df_clean = df.copy()

        logger.info(
            f"Running IQR outlier treatment "
            f"(multiplier={multiplier})..."
        )

        for col in cols_to_check:

            if (
                col not in df_clean.columns
                or not pd.api.types.is_numeric_dtype(df_clean[col])
            ):
                continue

            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)

            IQR = Q3 - Q1

            if IQR == 0 or pd.isna(IQR):
                continue

            lower_fence = Q1 - multiplier * IQR
            upper_fence = Q3 + multiplier * IQR

            outliers_count = (
                (df_clean[col] < lower_fence)
                | (df_clean[col] > upper_fence)
            ).sum()

            if outliers_count > 0:

                df_clean[col] = df_clean[col].clip(
                    lower=lower_fence,
                    upper=upper_fence
                )

                logger.info(
                    f"Clipped {outliers_count} IQR outliers "
                    f"in '{col}' to fence "
                    f"[{lower_fence:.2f}, {upper_fence:.2f}]."
                )

        return df_clean

    def detect_multivariate_isolation_forest(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:

        iso_cfg = self.config.get("isolation_forest", {})

        contamination = iso_cfg.get("contamination", 0.02)
        random_state = iso_cfg.get("random_state", 42)

        cols_to_check = [
            col
            for col in self.config.get(
                "numerical_columns_to_check", []
            )
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        df_clean = df.copy()

        if not cols_to_check:
            logger.warning(
                "No configured numerical columns found "
                "for Isolation Forest."
            )
            return df_clean

        logger.info(
            "Running Isolation Forest "
            f"(contamination={contamination})..."
        )

        # Select numerical features.
        X = df_clean[cols_to_check].replace(
            [np.inf, -np.inf], np.nan
        )

        # Isolation Forest requires finite input values.
        # Preserve the original row count by imputing missing
        # values with the median of each feature.
        X = X.fillna(X.median())

        # A completely missing feature still has a NaN median.
        # Remove such features from the model input.
        X = X.dropna(axis=1, how="all")

        if X.empty:
            logger.warning(
                "No usable numerical features for Isolation Forest."
            )
            return df_clean

        # Isolation Forest needs at least two samples.
        if len(X) < 2:
            logger.warning(
                "Insufficient rows for Isolation Forest."
            )
            return df_clean

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        preds = model.fit_predict(X)

        df_clean["is_multivariate_anomaly"] = (
            preds == -1
        ).astype(int)

        anomaly_count = df_clean[
            "is_multivariate_anomaly"
        ].sum()

        logger.info(
            f"Isolation Forest flagged {anomaly_count:,} "
            "multivariate anomalous records."
        )

        return df_clean


if __name__ == "__main__":

    config_file = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "O01_cleaning_rules.yaml"
    )

    sample_df = pd.DataFrame({
        "tenure_months": [12, 24, 6, 200, 18],
        "monthly_charges": [29.95, 75.0, 19.0, 550.0, 45.0],
        "total_charges": [359.4, 1800.0, 114.0, 99999.0, 810.0]
    })

    detector = OutlierDetector(config_path=config_file)

    df_z = detector.detect_and_treat_zscore(sample_df)

    df_iqr = detector.detect_and_treat_iqr(df_z)

    df_final = detector.detect_multivariate_isolation_forest(
        df_iqr
    )

    print("\n[RESULT] Outlier Treated DataFrame Preview:")
    print(df_final)

