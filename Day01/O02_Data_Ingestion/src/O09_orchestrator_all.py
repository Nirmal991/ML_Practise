import logging
from pathlib import Path
import pandas as pd

# Clean standard Python imports using the O-prefix convention
from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from extractors.O04_db_extractor import DatabaseExtractor
from extractors.O05_api_scraper import APIScraper
from extractors.O06_kafka_consumer import KafkaTelemetryConsumer
from hygiene.O07_schema_enforcer import SchemaEnforcer
from optimization.O08_memory_optimizer import MemoryOptimizer

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage2Orchestrator")

def get_file_batch_data(raw_dir: Path, filename: str = "customer_temetry.csv") -> pd.DataFrame:
    extractors = BatchFileExtractor(raw_data_dir=raw_dir)
    return extractors.extract_file(filename, file_name="csv")

def get_db_data() -> pd.DataFrame:
    db_extractor = DatabaseExtractor(connection_string="sqlite:///mock_enterprise.db")
    batch_generator = db_extractor.extract_in_batches("SELECT * FROM customer_telemetry", chunk_size=1000)
    return next(batch_generator)

def get_api_scrapper_data() -> pd.DataFrame:
    scraper = APIScraper(base_url="https://api.example-enterprise.com")
    payload = scraper.fetch_endpoint(endpoint="telemetry/recent", params={"limit":10}, rate_limit_delay=0.1)
    
    if payload and "data" in payload:
        df = pd.DataFrame(payload["data"])
        if "tenure_months" not in df.columns: df["tenure_months"] = 12
        if "monthly_charges" not in df.columns: df["monthly_charges"] = 49.99
        if "total_charges" not in df.columns: df["total_charges"] = 599.88
        if "contract_type" not in df.columns: df["contract_type"] = "month-to-month"
        if "payment_method" not in df.columns: df["payment_method"] = "credit_card"
        if "churn_status" not in df.columns: df["churn_status"] = "no"
        return df
    return pd.DataFrame()

def get_kafka_data() -> pd.DataFrame:
    """Extraction Channel 4: Real-time event streaming via Kafka consumer."""
    consumer = KafkaTelemetryConsumer(
        bootstrap_servers="localhost:9092",
        topic="customer-telemetry-events",
        group_id="ml-ingestion-group-v1"
    )
    messages = list(consumer.poll_messages(max_messages=5))
    if messages:
        df = pd.DataFrame(messages)
        # Ensure schema compliance mock fields
        if "total_charges" not in df.columns: df["total_charges"] = 250.0
        if "contract_type" not in df.columns: df["contract_type"] = "one-year"
        if "payment_method" not in df.columns: df["payment_method"] = "electronic_check"
        return df
    return pd.DataFrame()

# ==============================================================================
# MAIN PIPELINE RUNNER
# ==============================================================================


def run_stage2_pipeline():
    logger.info("==================================================")
    logger.info(" STARTING STAGE 2: INGESTION & HYGIENE PIPELINE")
    logger.info("==================================================")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    raw_dir = project_root / "data/raw"
    quarantine_dir = project_root / "data/quarantine"
    contract_path = project_root / "config/O01_schema_contract.yaml"
    
    raw_dir.mkdir(parents=True,exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    # Track active source for conditional validation and extraction
    # active_source = "file"  # Options: 'file', 'db', 'api', 'kafka'
    # active_source = "kafka"  # Set the active source for this pipeline run
    active_source = "db"  # Set the active source for this pipeline run
    
    logger.info(f"\n--- STEP A & B: Multi-Source Extraction [{active_source.upper()}] ---")
    
    if active_source == "file":
        target_file = raw_dir / "custom_telemetry.csv"
        
        if not target_file.exists():
            logger.info("Creating sample telemetry dataset for file batch execution...")
            target_file.write_text(
                "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
                "CUST-1001,12,29.95,359.4,month-to-month,electronic_check,no\n"
                "CUST-1002,24,75.50,1812.0,one-year,mailed_check,yes\n"
                "CUST-1003,-5,105.00,525.0,two-year,credit_card,no\n",  # tenure -5 fails contract
                encoding="utf-8"
            )
            
        # 2. File Integrity & Zero-Byte Check
        logger.info("Running File Integrity & Zero-Byte Check...")
        is_valid_file = validate_file_integrity(target_file)
        if not is_valid_file:
            logger.critical("Pipeline halted: File failed integrity checks.")
            return
        
        #3. Extract batch file data
        raw_df = get_file_batch_data(raw_dir=raw_dir, filename="customer_telemetry.csv")
        
    elif active_source == "db":
        raw_df = get_db_data()
        
    elif active_source == "api":
        raw_df = get_api_scrapper_data()
        
    elif active_source == "kafka":
        raw_df = get_kafka_data()
        
    else:
        raise ValueError(f"Unknown active extraction source: {active_source}")
    
    # Step C: Schema Contract Enforcement & Quarantine Routing
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine Routing ---")
    enforce = SchemaEnforcer(contract_path=contract_path, quarantine_dir=quarantine_dir)
    clean_df = enforce.validate_and_quarantine(raw_df, batch_id=f"batch_{active_source}_0O1")
    
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