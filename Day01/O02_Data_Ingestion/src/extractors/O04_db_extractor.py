# SQL/NoSQL connection pooling & chunking
import logging
from typing import Generator, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(ascii)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DatabaseExtractor")

class DatabaseExtractor:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        logging.info("Initialized DatabaseExtractor with connection target.")
        
    def extract_in_batches(
        self, query: str, chunk_size:int = 10000
    ) -> Generator[pd.DataFrame,None,None]:
        logger.info(f"Executing paginated query with chunk_size={chunk_size:,}...")
        
        logging.warning("[SIMULATION MODE] Database connection string not active. Yielding mock batch generator.")
        mock_data = pd.DataFrame({
            "customer_id": [f"CUST-{i:03d}" for i in range(1, 6)],
            "tenure_months": [12, 24, 6, 48, 3],
            "monthly_charges": [29.95, 75.50, 19.00, 105.20, 45.00],
            "churn_status": ["no", "yes", "no", "no", "yes"]
        })
        yield mock_data
        
if __name__ == "__main__":
    db_extractor = DatabaseExtractor(connection_string="sqlite:///mock_enterprise.db")

    batch_generator = db_extractor.extract_in_batches(
        query="SELECT * FROM customer_telemetry", chunk_size=2
    )

    for idx, batch_df in enumerate(batch_generator):
        logger.info(f"Processing database batch chunk #{idx + 1} with {len(batch_df)} rows.")
        print(batch_df)