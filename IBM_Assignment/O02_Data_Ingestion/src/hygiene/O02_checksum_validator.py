import hashlib
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ChecksumValidator")

def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_file_integrity(file_path: Path, expected_checksum: str = None) -> bool:
    
    #1. exist or not
    if not file_path.exists():
        logger.error(f"File not found in path: {file_path}")
        return False
    
    file_size_bytes = file_path.stat().st_size  # get the file size in bytes
    logger.info(f"Inspecting file: {file_path.name} | Size: {file_size_bytes} bytes")
    
    #2. zero byte check
    if file_size_bytes == 0:
        logger.critical(f"Zero-byte check FAILED: {file_path.name} is empty (0 bytes).")
        return False
    
    logger.info(f"Zero-byte check PASSED for {file_path.name}")
    
    #3. Optional checksum validation
    if expected_checksum:
        actual_checksum = calculate_sha256(file_path)
        logger.info(f"Computed SHA-256 : {actual_checksum}")
        logger.info(f"Expected SHA-256 : {expected_checksum}")
        
        if actual_checksum != expected_checksum:
            logger.critical(f"Checksum verification FAILED! Payload may be corrupted or tampered with.")
            return False
        
        logger.info("Checksum Verification PASSED")
    return True

def validate_payload_health(df: pd.DataFrame, source_name: str) -> str:
    
    logger.info(f"Running payload health check for source: [{source_name}]...")
    
    # 1. Emptyness check (equal to zero byte check in df)
    if df is None or df.empty:
        logger.critical(f"[HEALTH CHECK FAILED] Source '{source_name}' returned 0 records.")
        return False
    
    # 2. Volume Anomaly Check (e.g., abnormally low row count)
    min_expexted_row = 1
    if len(df) < min_expexted_row:
        logger.warning(f"[WARNING] Low volume warning from '{source_name}': Only {len(df)} records received.")
        
    # 3. Critical Column Presence Check
    essential_cols = ["customerID"]
    for col in essential_cols:
        if col not in df.columns:
            logger.critical(f"[HEALTH CHECK FAILED] Essential identifier column '{col}' missing from '{source_name}'.")
            return False
        
    logger.info(f"[HEALTH CHECK PASSED] Source '{source_name}' is healthy ({len(df):,} records loaded).")
    return True
    