import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DataLeakageValidator")

class DataLeakageValidator:
    def __init__(self, test_size = 0.2, random_state:int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    def stratified_train_test_split(self, df: pd.DataFrame, target_col: str) -> tuple:
        if target_col not in df.columns:
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
            return train_df, test_df
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[target_col]
        )
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)   
        overlap = train_indices & test_indices
        if overlap:
            raise ValueError(f"Data leakage detected: overlapping indices between train and test sets. : {overlap}")
        else:
            logger.info("Success. No overlapping indices detected between train and test sets.")
        
        return train_df, test_df

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "feature_1": range(100),
        "feature_2": range(100, 200),
        "churn_status": [0]*80 + [1]*20
    })
    validator = DataLeakageValidator(test_size=0.2, random_state=42)
    train_df, test_df = validator.stratified_train_test_split(sample_df, target_col="churn_status")
    print(f"Train DataFrame: {len(train_df)}")
    print(train_df.head())
    print(f"\nTest DataFrame: {len(test_df)}")
    print(test_df.head())