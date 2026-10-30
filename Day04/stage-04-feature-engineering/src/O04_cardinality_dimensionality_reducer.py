import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CardinalityReducer")

class CardinalityDimensionalityReducer:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.frequent_categories = {}
        self.pca_transformer = None
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("cardinality_reduction", {})
        
    def group_rare_categories(self, df: pd.DataFrame, categorical_cols: list, is_train: bool = True) -> pd.DataFrame:
        threshold = self.config.get("rare_label_threshold", 0.02)
        df_clean = df.copy()
        
        existing_col = [col for col in categorical_cols if col in df_clean.columns]
        if not existing_col:
            return df_clean
        
        if is_train:
            self.frequent_categories = {}
            for col in existing_col:
                frequencies = df_clean[col].value_counts(normalize=True)
                frequent = frequencies[frequencies >= threshold].index.tolist()
                self.frequent_categories[col] = frequent
                df_clean[col] = df_clean[col].apply(lambda x:x if x in frequent else "Other")
        else:
            if not self.frequent_categories:
                raise ValueError("[ERROR] Frequent categories not set. Run group_rare_categories with is_train=True first.")
            for col in existing_col:
                frequent = self.frequent_categories.get(col, [])
                df_clean[col] = df_clean[col].apply(lambda x: x if x in frequent else "Other")
                
        return df_clean
    
    def apply_pca_reduction(self, df: pd.DataFrame, variance_ratio: float = 0.95, is_train: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) <= 2:
            return df_clean
        
        if is_train:
            logger.info(f"Fitting PCA to retain {variance_ratio * 100}% cumulative explained variance...")
            self.pca_transformer = PCA(n_components=variance_ratio, random_state=42)
            pca_array = self.pca_transformer.fit_transform(df_clean[num_cols])
            
            pca_cols = [f"pc_{i+1}" for i in range(pca_array.shape[1])]
            pca_df = pd.DataFrame(pca_array, columns=pca_cols, index=df_clean.index)
            
            df_clean = pd.concat([df_clean.drop(columns=num_cols), pca_df], axis=1)
            logger.info(f"PCA reduced numerical feature space from {len(num_cols)} to {len(pca_cols)} components.")
            
        else:
            if self.pca_transformer is None:
                raise ValueError("PCA transformer not fitted. Run on train split first.")
            logger.info("Transforming test dataset using fitted PCA transformer...")
            
            pca_array = self.pca_transformer.transform(df_clean[num_cols])
            pca_cols = [f"pc_{i+1}" for i in range(pca_array.shape[1])]
            pca_df = pd.DataFrame(pca_array, columns=pca_cols, index=df_clean.index)
            
            df_clean = pd.concat([df_clean.drop(columns=num_cols), pca_df], axis=1)

        return df_clean
    
if __name__ == "__main__":
    config_file = Path(__file__).parent.parent / "config/O01_feature_rules.yaml"
    sample_df = pd.DataFrame({
        "payment_method": ["electronic_check", "mailed_check", "rare_method_1", "rare_method_2", "electronic_check"] + ["mailed_check"] * 95 ,
        "feature_1": [1.2, 2.3, 1.1, 4.5, 2.2] * 20,
        "feature_2": [3.4, 1.2, 3.5, 8.9, 2.1] * 20,
        "feature_3": [5.1, 2.3, 4.5, 7.8, 3.2] * 20
    })
    if config_file.exists():
        reducer = CardinalityDimensionalityReducer(config_path=config_file)
        df_reduced = reducer.group_rare_categories(sample_df, categorical_cols=["payment_method"], is_train=True)
        print("\n[RESULT] Rare Category Grouped DataFrame Preview:")
        print(df_reduced)
        df_reduced = reducer.apply_pca_reduction(df_reduced, variance_ratio=0.95, is_train=True)
        print("\n[RESULT] PCA Reduced DataFrame Preview:")
        print(df_reduced)