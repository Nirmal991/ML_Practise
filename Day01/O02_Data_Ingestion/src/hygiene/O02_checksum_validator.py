# File existence, size & hash validation

import hashlib
import logging
from pathlib import Path
import pandas as pd 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(message)s"
)

logger = logging.getLogger('ChecksumValidators')

def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_file_integrity(file_path: Path, expected_checksum: str = None) -> bool:
    
    if not file_path.exists():
        logger.error(f"File not found at path: {file_path}")
        return False
    
    file_size_bytes = file_path.stat().st_size
    logger.info(f"Inspecting file: {file_path.name} | Size: {file_size_bytes} bytes")
    
    if file_size_bytes == 0:
        logger.critical(f"Zero-byte check FAILED: {file_path.name} is empty (0 bytes).")
        return False
    
    logger.info(f"Zero-byte check Passed for {file_path.name}")
    
    if expected_checksum:
        actual_checksum = calculate_sha256(file_path)
        logger.info(f"Computed SHA-256: {actual_checksum}")
        logger.info(f"Expected SHA-256: {expected_checksum}")
        
        if expected_checksum != actual_checksum:
            logger.critical("checksum verification FAILED! Payload may be corrupted or tampeered with")
            return False
        
        logger.info("checksum verification PASSED.")
    return True

def validate_payload_health(df: pd.DataFrame, source_name: str) -> bool:
    
    logger.info(f"Running payload health check for source: [{source_name}]...")
    
    if df is None or df.empty:
        logger.critical(f"[HEALTH CHECK FAILED] Source'{source_name} returned 0 records.")
        return False
    
    min_expected_rows = 1
    if len(df) < min_expected_rows:
        logger.warning(f"[WARNING] Low volume warning from '{source_name}': Only {len(df)} records recieved.")
        
    essential_cols = ["customer_id"]
    for col in essential_cols:
        if col not in df.columns:
            logger.critical(f"[HEALTH CHECK FAILED] Essential identifier column '{col}' missing from '{source_name}'.")
            return False
        
    logger.info(f"[HEALTH CHECK PASSED] Source '{source_name}' is healthy ({len(df):,} records loaded).")
    return True


if __name__ == "__main__":
    print("Current working directory:", Path.cwd())
    sample_dir = Path("data/raw")
    print("Target directory:", sample_dir.resolve())
    
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = sample_dir / "test_telement.csv"
    
    if not test_file.exists():
        test_file.write_text(
            "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
            "CUST-001,12,29.95,359.4,month-to-month,electronic_check,no\n",
            encoding="utf-8"
        )
        
    print("File exists:", test_file.exists())
    print("File path:", test_file.resolve())
        
    is_valid = validate_file_integrity(test_file)
    print(f"\n[RESULT] File Integrity Status: {'PASSED (Ready for Ingestion)' if is_valid else 'REJECTED'}")
        
    
    