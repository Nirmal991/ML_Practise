import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("FeatureSelector")


"""feature_selection:
  variance_threshold:
    enabled: true
    threshold: 0.01 # Drop features with near-zero variance
  correlation_filter:
    enabled: true
    threshold: 0.85 # Drop one of any pair with absolute correlation > 0.85
  statistical_scoring:
    method: "mutual_info" # 'mutual_info' or 'chi2'
    top_k: 15"""
    
class FeatureSelector:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.variance_threshold = None
        self.selected_feature = None
        
    def _load_config(self):
        if not self.config_path:
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("feature_selection", {})
        
    def apply_variance_threshold(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        var_cfg = self.config.get("variance_threshold", {})
        if not var_cfg:
            return df
        
        threshold = var_cfg.get("threshold", 0.01)
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if not num_cols:
            return df_clean
        
        if is_train:
            self.variance_selector = VarianceThreshold(threshold=threshold) # examin and tell which one to keep
            self.variance_selector.fit(df_clean[num_cols]) #learn the feature of each data set which to keep and whic to not
            
            retrained_cols = [col for col, keep in zip(num_cols, self.variance_selector.get_support ()) if keep]
            dropped_cols = set(num_cols) - set(retrained_cols)
            
            if dropped_cols:
                logger.info(f"VarianceThreshold dropped low-variance columns: {list(dropped_cols)}")
            
            non_num_cols = [c for c in df_clean.columns if c not in num_cols]
            df_clean = pd.concat([df_clean[non_num_cols], df_clean[retrained_cols]], axis=1)
            
        else:
            if self.variance_selector is None:
                raise ValueError("VarianceThreshold selector not fitted. Run on train split first.")
            
            non_num_cols = [c for c in df_clean.columns if c not in num_cols]
            retained_cols = [col for col, keep in zip(num_cols, self.variance_selector.get_support()) if keep]
            df_clean = pd.concat([df_clean[non_num_cols], df_clean[retained_cols]], axis=1)
            
        return df_clean
    
    def apply_correleation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        corr_cfg = self.config.get("correlation_filter", {})
        if not corr_cfg.get("enable", True):
            return df
        
        threshold = corr_cfg.get("threshold", 0.85)
        df_clean = df.copy()
        num_df = df_clean.select_dtypes(include=[np.number])
        
        if num_df.empty or num_df.shape[1] <= 1:
            return df_clean
        
        corr_matrix = num_df.corr().abs()
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            df_clean.drop(columns=to_drop, inplace=True)

        return df_clean
    
    def apply_statistical_scoring(self, df: pd.DataFrame, target_col: str, is_train: bool = True) -> pd.DataFrame:
        stat_cfg = self.config.get("statistical_scoring", {})
        top_k = stat_cfg.get("top_k", 15)
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping statistical feature selection.")
            return df
        
        df_clean = df.copy()
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])]
        
        if len(feature_cols) <= top_k:
            logger.info("Feature count is less than or equal to top_k limit. Skipping subset truncation.")
            self.selected_features = feature_cols
            return df_clean
        
        if is_train:
            logger.info(f"Selecting top {top_k} features using Mutual Information classification scoring...")
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            selector  = SelectKBest(score_func=mutual_info_classif, k=top_k)
            selector.fit(X,y)
            
            self.selected_features = [col for col, keep in zip(feature_cols, selector.get_support()) if keep]
            logger.info(f"Selected top features: {self.selected_features}")
            
            non_feature_cols = [c for c in df_clean.columns if c not in feature_cols]
            df_clean = pd.concat([df_clean[non_feature_cols], df_clean[self.selected_features]], axis=1)
        else:
            if not self.selected_features:
                raise ValueError("Selected features not fitted. Run on train split first.")
            logger.info("Filtering test split to retain selected features...")
            non_feature_cols = [c for c in df_clean.columns if c not in feature_cols or c in self.selected_features]
            df_clean = df_clean[non_feature_cols]
            
        return df_clean
    
if __name__ == "__main__":
    config_file = Path("config/O01_feature_rules.yaml")

    sample_df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [1.01, 2.02, 3.01, 4.02, 5.01], # Highly collinear with feature_a
        "feature_c": [5.0, 4.0, 3.0, 2.0, 1.0],
        "constant_feature": [1.0, 1.0, 1.0, 1.0, 1.0], # Zero variance
        "churn_status": [0, 1, 0, 1, 0]
    })
    print("\n[INFO] sample DataFrame Preview")  
    print(sample_df.head())
    if config_file.exists():
        selector = FeatureSelector(config_path=config_file)
        df_var = selector.apply_variance_threshold(sample_df, is_train=True)
        df_corr = selector.apply_correleation_filter(df_var)
        df_final = selector.apply_statistical_scoring(df_corr, target_col="churn_status", is_train=True)
        print("\n[RESULT] Selected Features DataFrame Preview:")
        print(df_final.head())          
        
        