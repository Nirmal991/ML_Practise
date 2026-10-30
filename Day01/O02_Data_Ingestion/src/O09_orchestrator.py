# Main end-to-end pipeline execution runner

"""
================================================================================
MODULE: Stage 2 Master Pipeline Orchestrator
FILE: src/O09_orchestrator.py
PURPOSE:
  Serves as the main chronological execution runner for Stage 2. Integrates 
  all modular components into a cohesive workflow: running checksum validation, 
  landing zone extraction, contract enforcement with quarantine routing, and 
  memory-optimized type downcasting in correct sequential order.
================================================================================
"""

import logging
from pathlib import Path

# Clean standard Python imports using the O-prefix convention
from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from hygiene.O07_schema_enforcer import SchemaEnforcer
from optimization.O08_memory_optimizer import MemoryOptimizer

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage2Orchestrator")


def run_stage2_pipeline():
    """Executes the complete end-to-end Data Ingestion and Hygiene pipeline."""
    logger.info("==================================================")
    logger.info(" STARTING STAGE 2: INGESTION & HYGIENE PIPELINE")
    logger.info("==================================================")

    # Resolve paths relative to orchestrator location
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    raw_dir = project_root / "data/raw"
    quarantine_dir = project_root / "data/quarantine"
    contract_path = project_root / "config/O01_schema_contract.yaml"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    target_file = raw_dir / "customer_telemetry.csv"
    if not target_file.exists():
        logger.info("Creating sample telemetry dataset for pipeline execution...")
        target_file.write_text(
            "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
            "CUST-1001,12,29.95,359.4,month-to-month,electronic_check,no\n"
            "CUST-1002,24,75.50,1812.0,one-year,mailed_check,yes\n"
            "CUST-1003,-5,105.00,525.0,two-year,credit_card,no\n",  # tenure -5 fails contract
            encoding="utf-8"
        )

    # Step A: Checksum & Integrity Validation
    logger.info("\n--- STEP A: File Integrity & Zero-Byte Check ---")
    is_valid_file = validate_file_integrity(target_file)
    if not is_valid_file:
        logger.critical("Pipeline halted: File failed integrity checks.")
        return

    # Step B: Batch Extraction
    logger.info("\n--- STEP B: Landing Zone Batch Extraction ---")
    extractor = BatchFileExtractor(raw_data_dir=raw_dir)
    raw_df = extractor.extract_file("customer_telemetry.csv", file_format="csv")

    # Step C: Schema Contract Enforcement & Quarantine Routing
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine Routing ---")
    enforcer = SchemaEnforcer(contract_path=contract_path, quarantine_dir=quarantine_dir)
    clean_df = enforcer.validate_and_quarantine(raw_df, batch_id="batch_telemetry_001")

    # Step D: Low-Level Memory Optimization (Downcasting)
    logger.info("\n--- STEP D: Memory Optimization & Type Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)

    logger.info("==================================================")
    logger.info(" STAGE 2 COMPLETED SUCCESSFULLY!")
    logger.info(f" Final Clean Records Ready for Stage 3: {len(optimized_df):,}")
    logger.info("==================================================")
    
    print("\n[FINAL CLEAN & OPTIMIZED DATAFRAME PREVIEW]")
    print(optimized_df)

if __name__ == "__main__":
    run_stage2_pipeline()