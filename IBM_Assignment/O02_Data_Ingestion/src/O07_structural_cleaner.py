import logging
from pathlib import Path
import pandas as pd
import yaml

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("StructuralCleaner")

class StructuralCleaner:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cofig = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("structural_cleaning", {})
        
    def remove_duplicates(self, df: pd.DataFrame, subset_cols: list = None) -> pd.DataFrame:
        # df_clean = df.copy()
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count:,} duplicate records. Remaining rows: {len(df_clean):,}")
        else: 
            logger.info("No duplicate record found.")
            
        return df_clean
    
    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing string columns (whitespace stripping & casing)...")
        df_clean = df.copy()
        
        text_cols = self.config.get("text_columns", [])
        case_format = self.config.get("case_format", "lower")
        strip_ws = self.config.get("strip_whitespace", True)
        
        for col in text_cols:
            if col in df_clean.columns:
                series = df_clean[col].astype(str)
                
                if strip_ws:
                    series = series.str.strip()
                    
                if case_format == "lower":
                    series.str.lower()
                elif case_format == "upper":
                    series.str.upper()
                elif case_format == "title":
                    series.str.title()
                    
                df_clean[col] = series
                logger.info(f"Normalized text column '{col}' to {case_format} case with whitespace stripped.")
                
        return df_clean
    
    def apply_caanonical_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        
        fuzzy_cfg = self.config.get("fuzzy_matching", {})
        if not fuzzy_cfg.get("enabled", False):
            logger.info("Fuzzy matching is disabled in configuration.")
            return df
        
        cannonical_mapping = fuzzy_cfg.get("canonical_mappings", {})
        threshold = fuzzy_cfg.get("similareshity_throld", 85)
        
        df_clean = df.copy()
        logger.info(f"Applying canonical mappings and fuzzy matching (threshold >= {threshold})...")
        
        for col, cannonical_dict in cannonical_mapping.item():
            if col not in df_clean.columns:
                continue
            
            exact_lookup = {}
            for cannonical_val, variants in cannonical_dict.items():
                exact_lookup[cannonical_val.lower()] = cannonical_val()
                for variant in variants:
                    exact_lookup[variant.lower()] = cannonical_val