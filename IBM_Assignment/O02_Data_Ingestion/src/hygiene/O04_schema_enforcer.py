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
    def __init__(self, contract_path: Path, qurantine_dir: Path):
        self.contract_path = Path(contract_path)
        self.qurantine_dir = Path(qurantine_dir)
        self.qurantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.contract = self._load_contract()
        
    def _load_contract(self):
        if not self.contract_path.exists():
            raise FileNotFoundError(f"[ERROR] Schema contract not found: {self.contract_path}")
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)
            
        logger.info(f"Loaded schema contract: {contract_data['dataset_info']['name']} (v{contract_data['dataset_info']['version']})")
        return contract_data
    
    def validate_and_qurantine(self, df: pd.DataFrame, batch_id: str = "batch_001"):
        expected_cols = self.contract["expected_columns"]
        
        missing_columns = [col for col in expected_cols.keys() if col not in df.columns]
        if missing_columns:
            logger.error(f"[SCHEMA DRIFT] Missing expected columns in batch {batch_id}: {missing_columns}")
            qurantine_path = self.qurantine_dir / f"qurantined_{batch_id}_schema_drift.csv"
            raise ValueError(f"Schema drift detected! Entir batch qurantine to {self.qurantine_dir}")
        
        valid_indices = []
        qurantine_rows = []
        
        logger.info(f"Validating {len(df):,} rows against schema contract rules...")
        
        for idx,row in df.iterrows():
            is_row_valid = True
            error_reason = []
            
            for col,rules in expected_cols.items():
                val = row[col]
                
                # NUllability check
                if not rules.get("nullable", True) and pd.isna(val):  #??
                    is_row_valid = False
                    error_reason.append(f"Coulumn '{col}' cannot be null.")
                    break
                
                if pd.isna(val):
                    continue
                
                if "allowed_value" in rules and val not in rules["allowed_values"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value '{val}' not in allowed set {rules['allowed_values']}.")
                    break
                
                #Numeric range check
                if "min_val" in rules and val < rules["min_value"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value {val} below minimum {rules['min_value']}.")
                    break
                
                if "max_val" in rules and val > rules["max_value"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value {val} above maximum {rules['max_value']}.")
                    break
                
            if is_row_valid:
                valid_indices.append(idx)
            else:
                row_dict = row.to_dict()
                row_dict["quarantine_reason"] = " | ".join(error_reason)
                qurantine_rows.append(row_dict)
                
        # Route invalid row to Qurantine 
        if qurantine_rows:
            q_df = pd.DataFrame(qurantine_rows)
            q_file = self.qurantine_dir / f"quarantined_{batch_id}.csv"
            q_df.to_csv(q_file, index=False)
            logger.warning(f"[QUARANTINE] Routed {len(qurantine_rows):,} invalid rows to dead-letter queue: {q_file}")
            
        #4. filter and return valid dataframe
        clean_df = df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Validation complete. Passed: {len(clean_df):,} | Quarantined: {len(qurantine_rows):,}")
        
        return clean_df