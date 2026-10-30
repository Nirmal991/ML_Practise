# CSV, JSON, Parquet batch parsing
import logging 
from pathlib import Path
import pandas as pd

# configure logging
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BatchFileExtractor")


class BatchFileExtractor:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.exists():
            logging.warning(f"Raw data directory does not exist yet: {self.raw_data_dir}")
            
            
    def extract_file(self, file_name: str, file_format: str = "csv", **kwargs) -> pd.DataFrame:
        file_path = self.raw_data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"[ERROR] Target file not found: {file_path}")
        
        logger.info("Starting batch extraction for '{filename}' (Format: {file_format.upper()})...")
        
        try:
            if file_format.lower() == "csv":
                # Default to UTF-8, handling potential encoding issues gracefully
                df = pd.read_csv(file_path, encoding="utf-8", **kwargs)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: '{file_format}'. Use csv, json, or parquet.")
                
            logger.info(f"Successfully extracted {len(df):,} rows and {len(df.columns)} columns from {file_name}.")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_name}: {e}")
            raise e
        
        
if __name__ == "__main__":
    landing_zone = Path("data/raw")
    landing_zone.mkdir(parents=True, exist_ok=True)
    
    sample_csv = landing_zone / "customer_telementry.csv"
    
    # Create sample CSV if not present
    if not sample_csv.exists():
        sample_csv.write_text(
            "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
            "CUST-001,12,29.95,359.4,month-to-month,electronic_check,no\n"
            "CUST-002,34,56.50,1921.0,one-year,mailed_check,yes\n",
            encoding="utf-8"
        )
        
    extractor = BatchFileExtractor(raw_data_dir=landing_zone)
    df_extracted = extractor.extract_file("customer_telementry.csv", file_format="csv")
    print("\n[PREVIEW] Extracted Dataframe Head:")
    print(df_extracted.head())
            