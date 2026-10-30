"""
O07_orchestrator.py
Main master runner tying together all cleaning, sanitization, and EDA steps for Stage 3.
"""

import logging
from pathlib import Path
import pandas as pd

from O01_missingness_handler import MissingnessHandler
from O02_structural_cleaner import StructuralCleaner
from O03_outlier_detector import OutlierDetector
from O04_temporal_aligner import TemporalAligner
from O05_numerical_sanitizer import NumericalSanitizer
from O06_eda_profiler import EDAProfiler

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage3Orchestrator")


def run_stage3_pipeline():
    """Executes the complete end-to-end Data Preprocessing, Cleaning & EDA pipeline."""
    logger.info("==================================================")
    logger.info(" STARTING STAGE 3: PREPROCESSING, CLEANING & EDA")
    logger.info("==================================================")

        # 1. Resolve Paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    config_path = project_root / "config/O01_cleaning_rules.yaml"
    processed_dir = project_root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    input_file = project_root.parent / "data/raw/customer_telemetry.csv"
    if not input_file.exists():
        # Fallback inline sample dataset if Stage 2 file isn't found
        logger.info("file not found. Generating sample raw DataFrame for Stage 3 testing...")
        df = pd.DataFrame({
            "customer_id": ["CUST-001", "CUST-002", "CUST-002", "CUST-003"],
            "tenure_months": [12, -999.0, 24, 150],  # Includes MNAR sentinel & outlier
            "monthly_charges": [29.95, 75.0, 75.0, 550.0], # Includes duplicate row & outlier
            "total_charges": [359.4, None, 1800.0, 99999.0], # Includes missing & outlier
            "contract_type": [" month-to-month ", "m2m", "m2m", "1-year"],
            "payment_method": ["electronic_check", "mailed_check", "mailed_check", "credit_card"],
            "churn_status": ["no", "yes", "yes", "no"]
        })
    else:
        logger.info(f"Loading input dataset from Stage 2 path: {input_file}")
        df = pd.read_csv(input_file)

    logger.info(f"Initial raw dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2. Instantiate Cleaning Engines
    missingness_handler = MissingnessHandler(config_path=config_path)
    structural_cleaner = StructuralCleaner(config_path=config_path)
    outlier_detector = OutlierDetector(config_path=config_path)
    temporal_aligner = TemporalAligner(config_path=config_path)
    numerical_sanitizer = NumericalSanitizer(config_path=config_path)
    eda_profiler = EDAProfiler()


if __name__ == "__main__":
    run_stage3_pipeline()