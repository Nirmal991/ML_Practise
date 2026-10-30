import logging
from pathlib import Path

from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from hygiene.O04_schema_enforcer import SchemaEnforcer
from optimization.O05_memory_optimizer import MemoryOptimizer

from O06_missingness_handler import MissingnessHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage2Orchestrator")

def run_stage2_pipeline():
    logger.info("==================================================")
    logger.info(" STARTING STAGE 2: INGESTION & HYGIENE PIPELINE")
    logger.info("==================================================")
    
    # STEP 1: FILE HANDLING
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    raw_dir = project_root / "data/raw"
    qurantine_dir = project_root / "data/qurantine"
    contract_path = project_root / "config/O01_schema_contract.yaml"
    config_path = project_root / "config/O02_cleaning_rules.yaml"
    processed_dir = project_root / "data/processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    qurantine_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = raw_dir / "Telco-Customer-Churn.csv"
    if not target_file.exists():
        logger.warning("Dataset not found. Terminating pipeline.")
        return
    
    # STEP 1: CHECKSUM AND INTEGRITY VALIDATION
    logger.info("\n--- STEP A: File Integrity & Zero-Byte Check ---")
    is_valid_file = validate_file_integrity(target_file)
    if not is_valid_file:
        logger.critical("Pipeline halted: File failed integrity checks.")
        return
    
    # STEP 2: BATCH EXTRACTOR
    logger.info("\n--- STEP B: Landing Zone Batch Extraction ---")
    extractor = BatchFileExtractor(raw_data_file=raw_dir)
    raw_df = extractor.extract_file("Telco-Customer-Churn.csv", file_format="csv")
    
    # STEP 3: SCHEMA CONTRACT ENFORCMENT
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine Routing ---")
    enforce = SchemaEnforcer(contract_path=contract_path, qurantine_dir=qurantine_dir)
    clean_df = enforce.validate_and_qurantine(raw_df,batch_id="batch_telemetry_001")
    
    # Step 4: Low lwvwl MemoryOptimization (DownCasting)
    logger.info("\n--- STEP D: Memory Optimization & Type Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)
    
    logger.info("==================================================")
    logger.info(" STAGE 2 COMPLETED SUCCESSFULLY!")
    logger.info(f" Final Clean Records Ready for Stage 3: {len(optimized_df):,}")
    logger.info("==================================================")
    
    print("\n[FINAL CLEAN & OPTIMIZED DATAFRAME PREVIEW]")
    print(optimized_df.head())
    # print("Project root:", project_root)
    # print("First path:", project_root.parent / "data/raw/Telco-Customer-Churn.csv")
    # print("Second path:", project_root / "data/raw/Telco-Customer-Churn.csv")
    input_file = project_root / "data/raw/Telco-Customer-Churn.csv"
    if not input_file.exists():
        logger.info("file not found. Terminating the Code...")
        return
    
    logger.info(f"Initial raw dataset shape: {optimized_df.shape[0]:,} rows, {optimized_df.shape[1]} columns")
    
    #Step 5 Instantiate cleaning:
    missingness_handler = MissingnessHandler(config_path=config_path)
    
if __name__ == "__main__":
    run_stage2_pipeline()
    
    
    
    