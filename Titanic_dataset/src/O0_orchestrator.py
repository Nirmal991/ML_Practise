import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from hygiene.O02_checksum_validations import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from hygiene.O04_schema_enforcer import SchemaEnforcer
from optmization.O05_memory_optmizer import MemoryOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Orchestractor")

def run_pipeline():
    logger.info("=====STARTING THE PIPELINES WITH STAGE 2=====")
    
    # 1 File Handling:
    current_dir = Path(__file__).parent
    # print(current_dir)
    
    project_root = current_dir.parent
    # print(project_root)
    
    raw_dir = project_root / "data/raw"
    qurantine_dir = project_root / "data/qurantine"
    contract_path = project_root / "config/O01_schema_contract.yaml"
    
    raw_dir.mkdir(exist_ok=True, parents=True)
    qurantine_dir.mkdir(exist_ok=True, parents=True)
    
    target_file = raw_dir / "titanic_messy.csv"
    if not target_file.exists():
        logger.warning(f"File with this file Name not exists")
        return
    
    # STEP 1 CHECKSUM VALIDATIONS
    logger.info("\n--- STEP A: File Integrity & Zero-Byte Check ---")
    is_valid_file = validate_file_integrity(target_file)
    if not is_valid_file:
        logger.critical("Pipeline halted: File failed integrity checks.")
        return
    
    # STEP 2 BATCH EXTRACTOR:
    logger.info("\n--- STEP B: Landing Zone Batch Extraction ---")
    extractor = BatchFileExtractor(raw_data_file=raw_dir)
    raw_df = extractor.extract_file("titanic_messy.csv", file_format="csv")
    # print(raw_df.head())
    # print(raw_df.dtypes)
    
    raw_df["Age"] = pd.to_numeric(raw_df["Age"], errors="coerce")
    
    # STEP 3: SCHEMA CONTRACT ENFORCMENT
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine Routing ---")
    enforce = SchemaEnforcer(contract_path=contract_path, qurantine_dir=qurantine_dir)
    clean_df = enforce.validate_and_quarantine(raw_df, batch_id="batch_titanic_001")
    # print(clean_df.head())
    
    # Step 4: Low level MemoryOptimization (DownCasting)
    logger.info("\n--- STEP D: Memory Optimization & Type Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)
    # print(optimized_df.dtypes)
    
    logger.info("==================================================")
    logger.info(" STAGE 2 COMPLETED SUCCESSFULLY!")
    logger.info(f" Final Clean Records Ready for Stage 3: {len(optimized_df):,}")
    logger.info("==================================================")
    
    print("\n[FINAL CLEAN & OPTIMIZED DATAFRAME PREVIEW]")
    print(optimized_df.head())
    
    logger.info("===========STAGE 3 STARTING THE PREVIEW==============")
    
    config_path3 = project_root / "config/O06_cleaning_rules.yaml"
    processed_dir = project_root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    input_file_stage2 = project_root / "data/raw/titanic_messy.csv"
    if not input_file_stage2.exists():
        logger.info("file not found. Terminating the Code...")
        return 
    
    logger.info(f"Initial raw dataset shape: {optimized_df.shape[0]:,} rows, {optimized_df.shape[1]} columns")
    
    
    
if __name__ == "__main__":
    run_pipeline()