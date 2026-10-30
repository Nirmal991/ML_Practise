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
    def __init__(self, config_path:Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("structural_cleaning", {})
        
    
    def remove_duplicates(self, df: pd.DataFrame, subset_cols: list = None) -> pd.DataFrame:
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
        removed_count = initial_len - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count:,} duplicate records. Remaining rows: {len(df_clean):,}")
        else:
            logger.info("No duplicate record found.")
            
        return df_clean
    
    def normaliza_text(self, df:pd.DataFrame) -> pd.DataFrame:
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
                    series = series.str.lower()
                elif case_format == "upper":
                    series = series.str.upper()
                elif case_format == "title":
                    series = series.str.title()
                    
                df_clean[col] = series
                logger.info(f"Normalized text column '{col}' to {case_format} case with whitespace stripped.")\
                    
        return df_clean
    
    def apply_cannonical_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        fuzzy_cfg = self.config.get("fuzzy_matching", {})
        if not fuzzy_cfg.get("enable", False):
            logger.info("Fuzzy matching is disabled in configuration.")
            return df
        
        canonical_mappings = fuzzy_cfg.get("canonical_mappings", {})
        threshold = fuzzy_cfg.get("similarity_threshold", 85)
        
        df_clean = df.copy()
        logger.info(f"Applying canonical mappings and fuzzy matching (threshold >= {threshold})...")
        
        for col, canonical_dict in canonical_mappings.items():
            if col not in df_clean.columns:
                continue
            
            exact_lookup = {}
            for cannonical_val, variants in canonical_dict.items():
                exact_lookup[cannonical_val.lower()] == cannonical_val
                for variant in variants:
                    exact_lookup[variant.lower()] = cannonical_val
                    
            def map_value(val):
                if pd.isna(val):
                    return val
                val_str = str(val).lower().strip()
                
                # 1. Check exact lookup first
                if val_str in exact_lookup:
                    return exact_lookup[val_str]
                
                # 2. If not found and rapidfuzz is available, try fuzzy matching
                if RAPIDFUZZ_AVAILABLE:
                    all_variants = list(exact_lookup.keys())
                    match = process.extractOne(val_str, all_variants, scorer=fuzz.ratio)
                    if match and match[1] >= threshold:
                        matched_variant = match[0]
                        return exact_lookup[matched_variant]
                        
                # Fallback to original value if no match exceeds threshold
                return val
            
            df_clean[col] = df_clean[col].apply(map_value)
            logger.info(f"Standardized categorical column '{col}' using canonical mappings.")
            
        return df_clean
    
if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_cleaning_rules.yaml")
    
    sample_df = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-001", "CUST-002"], # Duplicate row
        "contract_type": [" month-to-month ", "m2m", "1-year"], # Messy formatting & typos
        "payment_method": ["electronic_check", "electronic_check", "mailed_check"]
    })
    
    if config_file.exists():
        cleaner = StructuralCleaner(config_path=config_file)
        df_dedup = cleaner.remove_duplicates(sample_df, subset_cols=["customer_id"])
        df_norm = cleaner.normaliza_text(df_dedup)
        df_final = cleaner.apply_cannonical_mapping(df_norm)
        print("\n[RESULT] Structurally Cleaned DataFrame:")
        print(df_final)