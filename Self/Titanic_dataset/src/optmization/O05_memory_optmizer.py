import logging
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
) 
logger = logging.getLogger("MemoryOptimizer")

class MemoryOptimizer:
    @staticmethod
    def get_Memory_usage_mb(df: pd.DataFrame) -> pd.DataFrame:
        return df.memory_usage(deep=True).sum() / (1024*1024)
    
    @classmethod
    def downcast_numeric_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        start_mem = cls.get_Memory_usage_mb(df)
        logger.info(f"Memory usage before downcasting: {start_mem:.2f} MB")
        
        for col in df.columns:
            col_types = df[col].dtype
            
            if 'int' in str(col_types):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max <  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
                    
            if 'float' in str(col_types):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.int64)
                    
        end_mem = cls.get_Memory_usage_mb(df)
        reduction = 100 * (start_mem-end_mem) / start_mem if start_mem > 0 else 0
        logger.info(f"Memory usage after downcasting : {end_mem:.2f} MB (Saved {reduction:.1f}% RAM)")
        return df
    
    @staticmethod
    def process_out_of_core_polars(file_path: Path) -> pd.DataFrame:
        logger.info(f"Scanning file lazily with Polars: {file_path.name}")
        try:
            lazy_df = pl.scan_csv(str(file_path))
            df = lazy_df.collect().to_pandas()
            return df
        except Exception as e:
            logger.warning(f"Polars engine execution skipped/unavailable: {e}. Falling back to pandas.")
            return pd.read_csv(file_path)
            