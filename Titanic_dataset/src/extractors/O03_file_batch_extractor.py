import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s [%(levelname)s %(name)s: %(messsage)s]"
)
logger = logging.getLogger("BatchFileExtractor")

class BatchFileExtractor:
    def __init__(self, raw_data_file: Path):
        self.raw_data_file = Path(raw_data_file)
        if not self.raw_data_file.exists():
            logger.warning(f"Raw data directory does not exist yet: {self.raw_data_dir}")
            
    def extract_file(self, file_name: str, file_format: str = "csv", **kwargs) -> pd.DataFrame:
        file_path = self.raw_data_file / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"[ERROR] Target file not found: {file_path}")
        
        logger.info(f"Starting batch extraction for '{file_name}' (Format: {file_format.upper()})...")
        
        try:
            if file_format.lower() == "csv":
                df = pd.read_csv(file_path, encoding="utf-8", **kwargs)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else: 
                raise ValueError(f"Unsupported File Format: '{file_format}'. Use csv, json, or parquet.")
            
            logger.info(f"Successfully extracted {len(df):,} rows and {len(df.columns)} columns from {file_name}.")
            return df
        except Exception as e:
            logger.error(f"failed to parse file {file_name}: {e}")
            raise e
            