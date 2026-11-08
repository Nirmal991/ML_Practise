import logging 
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import yaml
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SchemaEnforcer")

class SchemaEnforcer:
    def __init__(self, contract_path: Path, qurantine_dir: Path):
        self.contract_path = Path(contract_path)
        self.qurantine_dir = Path(qurantine_dir)
        self.qurantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.contrct = self._load_contract()
        self.expected_columns = self.contrct.get("expected_columns", {})
        
        if not isinstance(self.expected_columns, dict):
            raise ValueError("'expexted_columns' must be a YAML mapping.")
        
    def _load_contract(self):
        if not self.contract_path.exists():
            raise FileNotFoundError(f"Schema contract not found: {self.contract_path}")
        
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)
            
        if "dataset_info" not in contract_data:
            raise ValueError("YAML contract is missing 'dataset_info'.")
        if "expected_columns" not in contract_data:
            raise ValueError("YAML contract is missing 'expected_columns'.")

        info = contract_data["dataset_info"]
        logger.info(
            "Loaded schema contract: %s (v%s)",
            info.get("name", "Unknown"),
            info.get("version", "Unknown"),
        )
        return contract_data
    
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        normalized = []
        
        for column in df.columns:
            name = str(column).strip()
            name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
            name = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", name)
            name = re.sub(r"[^A-Za-z0-9]+", "_", name)
            name = re.sub(r"_+","_",name).strip("_").lower()
            normalized.append(name)
            
        if len(normalized) != len(set(normalized)):
            duplicates = pd.Index(normalized)[
                pd.Index(normalized).duplicated()
            ].unique().tolist()
            raise ValueError(
                f"Column standardization created duplicate names: {duplicates}"
            )
        df.columns = normalized
        logger.info("Standardized column names to snake_case.")
        return df
    
    @staticmethod
    def _matches_type(value: Any, expected_type: str) -> bool:
        expected_type = str(expected_type).lower()

        if expected_type in {"integer", "int"}:
            try:
                return float(value).is_integer()
            except (TypeError, ValueError, OverflowError):
                return False

        if expected_type in {"float", "number", "numeric"}:
            return (
                pd.api.types.is_number(value)
                and not isinstance(value, bool)
            )

        if expected_type in {"string", "str", "object"}:
            return isinstance(value, str)

        if expected_type in {"boolean", "bool"}:
            return isinstance(value, bool)

        raise ValueError(f"Unsupported type in YAML contract: {expected_type}")
    
    def _save_quarantine(
        self, rows: pd.DataFrame, batch_id: str, suffix: str = ""
    ) -> Path:
        path = self.qurantine_dir / f"qurantined_{batch_id}{suffix}.csv"
        rows.to_csv(path,index=False)
        return path
    
    def validate_and_quarantine(
        self, df: pd.DataFrame, batch_id: str = "batch_001"
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Structural column naming is performed before schema validation.
        df = self.normalize_column_names(df)
        expected_cols = self.expected_columns

        # Missing required columns indicate batch-level schema drift.
        missing_columns = [
            col for col in expected_cols if col not in df.columns
        ]

        if missing_columns:
            drift_df = df.copy()
            drift_df["quarantine_reason"] = (
                "Schema drift: missing expected columns: "
                + ", ".join(missing_columns)
            )
            path = self._save_quarantine(
                drift_df, batch_id, suffix="_schema_drift"
            )
            logger.error(
                "Schema drift: missing %s. Batch saved to %s",
                missing_columns,
                path,
            )
            raise ValueError(f"Entire batch quarantined due to schema drift: {path}")

        logger.info("Validating %s rows...", f"{len(df):,}")
        errors_by_index: Dict[Any, List[str]] = {
            idx: [] for idx in df.index
        }

        # Uniqueness is checked within the current batch.
        for col, rules in expected_cols.items():
            if rules.get("unique", False):
                duplicate_mask = df[col].notna() & df[col].duplicated(keep=False)
                for idx in df.index[duplicate_mask]:
                    errors_by_index[idx].append(
                        f"'{col}' must be unique; duplicate value '{df.at[idx, col]}'."
                    )

        for idx, row in df.iterrows():
            for col, rules in expected_cols.items():
                value = row[col]

                if pd.isna(value):
                    if not rules.get("nullable", True):
                        errors_by_index[idx].append(
                            f"Column '{col}' cannot be null."
                        )
                    continue

                expected_type = rules.get("type")
                if expected_type and not self._matches_type(value, expected_type):
                    errors_by_index[idx].append(
                        f"Column '{col}' expected {expected_type}, got "
                        f"'{value}' ({type(value).__name__})."
                    )
                    continue

                # Supports both spellings, though YAML should use allowable_values.
                allowed = rules.get(
                    "allowable_values", rules.get("allowed_values")
                )
                if allowed is not None and value not in allowed:
                    errors_by_index[idx].append(
                        f"Column '{col}' value '{value}' not in {allowed}."
                    )

                if "min_value" in rules:
                    if value < rules["min_value"]:
                        errors_by_index[idx].append(
                            f"Column '{col}' value {value} is below "
                            f"{rules['min_value']}."
                        )

                if "max_value" in rules:
                    if value > rules["max_value"]:
                        errors_by_index[idx].append(
                            f"Column '{col}' value {value} is above "
                            f"{rules['max_value']}."
                        )

        invalid_indices = [
            idx for idx, errors in errors_by_index.items() if errors
        ]
        valid_indices = [
            idx for idx, errors in errors_by_index.items() if not errors
        ]

        if invalid_indices:
            quarantine_df = df.loc[invalid_indices].copy()
            quarantine_df["quarantine_reason"] = [
                " | ".join(errors_by_index[idx]) for idx in invalid_indices
            ]
            path = self._save_quarantine(quarantine_df, batch_id)
            logger.warning(
                "Quarantined %s invalid rows to %s",
                f"{len(quarantine_df):,}",
                path,
            )

        clean_df = df.loc[valid_indices].reset_index(drop=True)
        logger.info(
            "Validation complete. Passed: %s | Quarantined: %s",
            f"{len(clean_df):,}",
            f"{len(invalid_indices):,}",
        )
        return clean_df
    
                