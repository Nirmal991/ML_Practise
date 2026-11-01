import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

'''
categorical_encoding:
  low_cardinality_columns:
    - "gender"
    - "Partner"
    - "Dependents"
    - "PhoneService"
    - "MultipleLines"
    - "InternetService"
    - "OnlineSecurity"
    - "OnlineBackup"
    - "DeviceProtection"
    - "TechSupport"
    - "StreamingTV"
    - "StreamingMovies"
    - "Contract"
    - "PaperlessBilling"
    - "PaymentMethod"
  encoding_method_low: "one_hot" # 'one_hot' or 'ordinal'
  drop_first: true
  encoding_method_low: "one_hot" # 'one_hot' or 'ordinal'
  drop_first: true

  high_cardinality_columns: [] # Use target encoding if any exist
  target_encoding:
    smoothing: 10.0
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CategoricalEncoder")

class CategoricalEncoder:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ohe_encoder = None
        self.target_encoding_maps = {}
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(F"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("categorical_encoding", {})
        
    def fit_transform_low_cardinality(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        low_card_cols = self.config.get("low_cardinality_columns", [])
        existing_cols = [col for col in low_card_cols if col in df.columns]
        if not existing_cols:
            logger.warning("[WARNING] No low cardinality columns found in the DataFrame for encoding.")
            return df
        df_clean = df.copy()
        drop_first = self.config.get("drop_first", True)
        
        if is_train:
            self.ohe_encoder = OneHotEncoder(  
                drop='first' if drop_first else None,# Controls whether the first category should be dropped(dummy variable trap).
                sparse_output=False, # By default, OneHotEncoder can return a sparse matrix, False->NUmpy array
                handle_unknown='ignore' # Handles categories that were not present during training. (Not show the error in the pipelines)
            )
            encoded_array = self.ohe_encoder.fit_transform(df_clean[existing_cols])
            encoded_cols = self.ohe_encoder.get_feature_names_out(existing_cols) # Names of the newly generated columns
            logger.info(f"[INFO] OneHotEncoder fitted on columns: {existing_cols}")
        else:
            if self.ohe_encoder is None:
                raise ValueError(f"[ERROR] OneHotEncoder has not been fitted yet. run train first.")
            
            encoded_array = self.ohe_encoder.transform(df_clean[existing_cols])
            encoded_cols = self.ohe_encoder.get_feature_names_out(existing_cols)
            logger.info(f"[INFO] OneHotEncoder applied on columns: {existing_cols}")
            
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols,index=df_clean.index)
        df_clean = pd.concat([df_clean.drop(columns=existing_cols), encoded_df], axis=1)
        logger.info(f"[INFO] One-hot encoding completed. New shape: {df_clean.shape}")
        return df_clean
    
    def fit_trandform_target_encoding(self, df: pd.DataFrame, target_col:str, is_train:bool = True) -> pd.DataFrame:
        high_cols = self.config.get("high_cardinality_columns", [])
        existing_cols = [col for col in high_cols if col in df.columns]
        
        if not existing_cols or target_col not in df.columns:
            return df
        
        df_clean = df.copy()
        smoothing = self.config.get("target_encoding", {}).get("smoothing", 10.0)
        global_mean = df_clean[target_col].mean()
        
        if is_train:
            logger.info(f"Fitting Target Encoder on columns: {existing_cols} with smoothing={smoothing}")
            self.target_encoding_maps = {}
            
            for col in existing_cols:
                stats = df_clean.groupby(col)[target_col].agg(["count", "mean"])
                # Apply smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
                smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
                self.target_encoding_maps[col] = smoothed.to_dict()
                
                df_clean[f"{col}_encoded"] = df_clean[col].map(self.target_encoding_maps[col]).fillna(global_mean)
                df_clean.drop(columns=[col], inplace=True)
                logger.info(f"Target encoding applied to column: {col}. New column: {col}_encoded")
        else: 
            if not self.target_encoding_maps:
                raise ValueError("Target encoder maps not fitted. Run on train split first.")
            logger.info("Transforming high-cardinality columns using fitted Target Encoder maps...")
            
            for col in existing_cols:
                mapping = self.target_encoding_maps.get(col, {})
                df_clean[f"{col}_encoded"] = df_clean[col].map(mapping).fillna(global_mean)
                df_clean.drop(columns=[col], inplace=True)
            logger.info(f"Target encoding applied to columns: {existing_cols}. New columns: {[f'{col}_encoded' for col in existing_cols]}")
        return df_clean
        
        