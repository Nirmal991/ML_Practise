
# Contract validation & quarantine routing

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("SchemaEnforcer")


class SchemaEnforcer:
    def __init__(self, contract_path: Path, quarantine_dir: Path):
        self.contract_path = Path(contract_path)
        self.quarantine_dir = Path(quarantine_dir)

        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.contract = self.load_contract()

    def load_contract(self) -> dict:
        if not self.contract_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Schema contract not found: {self.contract_path}"
            )

        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)

        logger.info(
            f"Loaded schema contract: "
            f"{contract_data['dataset_info']['name']} "
            f"(v{contract_data['dataset_info']['version']})"
        )

        return contract_data

    def validate_and_quarantine(
        self,
        df: pd.DataFrame,
        batch_id: str = "batch_001" # why we use batch here
    ) -> pd.DataFrame:

        expected_cols = self.contract["expected_columns"]

        missing_columns = [
            col for col in expected_cols.keys()
            if col not in df.columns
        ]

        if missing_columns:
            logger.error(
                f"[SCHEMA DRIFT] Missing expected columns in batch "
                f"{batch_id}: {missing_columns}"
            )

            quarantine_path = (
                self.quarantine_dir
                / f"quarantined_{batch_id}_schema_drift.csv"
            )

            df.to_csv(quarantine_path, index=False)

            raise ValueError(
                f"Schema drift detected! Entire batch quarantined to "
                f"{quarantine_path}"
            )

        valid_indices = []
        quarantine_rows = []

        logger.info(
            f"Validating {len(df):,} rows against schema contract rules..."
        )

        for idx, row in df.iterrows():
            is_row_valid = True
            error_reason = []

            for col, rules in expected_cols.items():
                val = row[col]

                # Check for null values
                if not rules.get("nullable", True) and pd.isna(val):
                    is_row_valid = False
                    error_reason.append(
                        f"Column '{col}' cannot be null."
                    )
                    break

                if pd.isna(val):
                    continue

                # Check allowed values
                if (
                    "allowed_values" in rules
                    and val not in rules["allowed_values"]
                ):
                    is_row_valid = False
                    error_reason.append(
                        f"Column '{col}' value '{val}' not in allowed "
                        f"set {rules['allowed_values']}."
                    )
                    break

                # Check minimum value
                if "min_value" in rules and val < rules["min_value"]:
                    is_row_valid = False
                    error_reason.append(
                        f"Column '{col}' value {val} below minimum "
                        f"{rules['min_value']}."
                    )
                    break

                # Check maximum value
                if "max_value" in rules and val > rules["max_value"]:
                    is_row_valid = False
                    error_reason.append(
                        f"Column '{col}' value {val} above maximum "
                        f"{rules['max_value']}."
                    )
                    break

            if is_row_valid:
                valid_indices.append(idx)
            else:
                row_dict = row.to_dict()
                row_dict["quarantine_reason"] = " | ".join(error_reason)
                quarantine_rows.append(row_dict)

        # Save invalid rows to quarantine
        if quarantine_rows:
            q_df = pd.DataFrame(quarantine_rows)

            q_file = (
                self.quarantine_dir / f"quarantined_{batch_id}.csv"
            )

            q_df.to_csv(q_file, index=False)

            logger.warning(
                f"[QUARANTINE] Routed {len(quarantine_rows):,} "
                f"invalid rows to dead-letter queue: {q_file}"
            )

        # Return valid rows
        clean_df = df.loc[valid_indices].reset_index(drop=True)

        logger.info(
            f"Validation complete. Passed: {len(clean_df):,} | "
            f"Quarantined: {len(quarantine_rows):,}"
        )

        return clean_df


if __name__ == "__main__":
    contract_file = Path("config/01_schema_contract.yaml")
    quarantine_folder = Path("data/qurantine")

    # Create the schema contract if it does not exist
    if not contract_file.exists():
        contract_file.parent.mkdir(parents=True, exist_ok=True)

        contract_file.write_text(
            "dataset_info:\n"
            "  name: 'customer_subscription_telemetry'\n"
            "  version: '1.0.0'\n"
            "expected_columns:\n"
            "  customer_id:\n"
            "    type: 'string'\n"
            "    nullable: false\n"
            "  tenure_months:\n"
            "    type: 'int64'\n"
            "    nullable: false\n"
            "    min_value: 0\n"
            "    max_value: 120\n",
            encoding="utf-8"
        )

    # Create mock test DataFrame with some invalid records
    test_data = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002", None],
        "tenure_months": [12, 150, 6],
        # 150 violates max_value of 120
    })

    # Initialize the schema enforcer
    enforcer = SchemaEnforcer(
        contract_path=contract_file,
        quarantine_dir=quarantine_folder
    )

    # Validate the data and quarantine invalid rows
    clean_data = enforcer.validate_and_quarantine(
        test_data,
        batch_id="test_batch_01"
    )

    print("\n[RESULT] Clean DataFrame output:")
    print(clean_data)
