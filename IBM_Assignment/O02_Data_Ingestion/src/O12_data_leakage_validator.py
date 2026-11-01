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
    def __init__(self, test_size:float = 0.2, random_state:int = 42):
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
            stratify=df[target_col] #The stratify parameter is used to maintain the same proportion of target classes in both training and testing datasets.
        )
        # train_df.index returns the row labels (index values) of the Pandas DataFrame.[random split index we will geteg, 3,6,8,1]
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        overlap = train_indices & test_indices # This calculates the common elements (intersection) between training and testing indices.
        if overlap:
            raise ValueError(f"Data leakage detected: overlapping indices between train and test sets. : {overlap}")
        else:
            logger.info("Success. No overlapping indices detected between train and test sets.")
            
        return train_df, test_df