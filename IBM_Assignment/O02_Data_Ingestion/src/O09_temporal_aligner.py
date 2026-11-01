import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TemporalAligner")

class TemporalAligner:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}]")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("temporal_alignment", {})
        
    def align_timezone_to_utc(self, df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
        target_tz = self.config.get("target_timezone", "UTC")
        df_clean = df.copy()
        
        logger.info(f"Normalizing datetime columns to target timezone: {target_tz}...")
        
        for col in datetime_cols:
            if col not in df_clean.columns:
                continue
            
            dt_series = pd.to_datetime(df_clean[col], errors="coerce")
            
            if dt_series.dt.tz is None:
                dt_series = dt_series.dt.localize(target_tz, nonexistent="shift_forward", ambiguous="NaT") if hasattr(dt_series.dt, "localize") else dt_series.dt.tz_localize(target_tz, ambiguous="NaT", nonexistent="shift_forward")
            else:
                dt_series = dt_series.dt.tz_convert(target_tz)
                
            df_clean[col] = dt_series
            logger.info(f"Successfully normalized '{col}' to {target_tz}.")
            
        return df_clean
    
    def resample_time_series(self, df: pd.DataFrame, datetime_col: str, value_cols: list) -> pd.DataFrame:
        if datetime_col not in df.columns:
            logger.warning(f"Datetime column '{datetime_col}' not found for resampling.")
            return df
        
        freq = self.config.get("resample_frequency", "1D")
        fill_strategy = self.config.get("fill_strategy", "ffill")
        
        df_clean = df.copy()
        logger.info(f"Resampling time-series on '{datetime_col}' at frequency '{freq}' (Strategy: {fill_strategy})...")
        
        df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
        df_clean = df_clean.sort_values(by=datetime_col)
        df_clean.set_index(datetime_col, inplace=True)
        
        existing_value_cols = [c for c in value_cols if c in df_clean.columns]
        if not existing_value_cols:
            return df_clean.reset_index()
        
        ressampled = df_clean[existing_value_cols].resample(freq).mean()
        
        if fill_strategy == 'ffill':
            ressampled = ressampled.ffill()
        elif fill_strategy == "bfill":
            ressampled = ressampled.bfill()
        else: 
            ressampled = ressampled.fillna(0)
            
        logger.info("Time-series resampling completed successfully.")
        return ressampled.reset_index()