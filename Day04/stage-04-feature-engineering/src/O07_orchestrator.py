import logging
from pathlib import Path
import pandas as pd
import joblib

from O01_domain_feature_builder import DomainFeatureBuilder
from O06_data_leakage_validator import DataLeakageValidator
from O04_cardinality_dimensionality_reducer import CardinalityDimensionalityReducer
from O02_categorical_encoder import CategoricalEncoder
from O03_numerical_transformer import NumericalTransformer
from O05_feature_selector import FeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage4Orchestrator")


def run_stage4_pipeline():
    # 1. Resolve Paths
    current_dir = Path(__name__).parent
    project_root = current_dir.parent
    
    config_path = project_root / "config/O01_feature_rules.yaml"
    models_dir = project_root / "models"
    processed_dir = project_root / "data/processed"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = project_root / "data/raw/input.csv"
    if input_file.exists():
        df = pd.read_csv(input_file)
    else:
        df = pd.DataFrame({
            "customer_id": [f"CUST-{i:03d}" for i in range(1, 101)],
            "tenure_months": [12, 24, 6, 48, 3, 18, 36, 12, 60, 2] * 10,
            "monthly_charges": [29.95, 75.0, 19.0, 105.0, 45.0, 80.0, 95.0, 30.0, 110.0, 20.0] * 10,
            "total_charges": [359.4, 1800.0, 114.0, 5040.0, 135.0, 1440.0, 3420.0, 360.0, 6600.0, 40.0] * 10,
            "contract_type": ["month-to-month", "one-year", "month-to-month", "two-year", "month-to-month"] * 20,
            "payment_method": ["electronic_check", "mailed_check", "credit_card", "bank_transfer", "mailed_check"] * 20,
            "churn_status": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 10
        })
        
        target_col = "churn_status"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing from input dataset.")
        
        # 2 step 1 : Strict Train/Test Split (Preventing Global Data Leakage)
        leakage_validator = DataLeakageValidator(test_size=0.20, random_state=42)
        train_df, test_df = leakage_validator.stratified_train_test_split(df, target_col)
        
        # 3. Instantiate Feature Engineering Engines
        domain_builder = DomainFeatureBuilder(config_path=config_path)
        cardinality_reducer = CardinalityDimensionalityReducer(config_path=config_path)
        cat_encoder = CategoricalEncoder(config_path=config_path)
        num_transformer = NumericalTransformer(config_path=config_path)
        feature_selector = FeatureSelector(config_path=config_path)
        
        # 4. STEP 2: Domain Feature Building (Applied independently to train and test)
        logger.info("\n--- STEP 2: Domain Feature Construction ---")
        train_df = domain_builder.build_features(train_df)
        test_df = domain_builder.build_features(test_df)
        
        # 5. STEP 3: Cardinality Reduction (Fit on train, transform test)
        cat_cols_to_reduce = ["payment_method"]
        train_df = cardinality_reducer.group_rare_categories(train_df, categorical_cols= cat_cols_to_reduce,is_train=True)
        test_df = cardinality_reducer.group_rare_categories(test_df, categorical_cols= cat_cols_to_reduce,is_train=False)
        
        # 6. STEP 4: Categorical Encoding (Fit on train, transform test)
        train_df = cat_encoder.fit_transform_low_cardinality(train_df, is_train=True)
        test_df = cat_encoder.fit_transform_low_cardinality(test_df, is_train=False)
        
        train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0.0)
        test_df[target_col] = test_df[target_col].fillna(test_df[target_col].mode()[0] if not test_df[target_col].empty else 0)
        
        # 8. STEP 6: Multistage Feature Selection (Fit on train, filter test)
        train_df = feature_selector.apply_variance_threshold(train_df, is_train=True)
        test_df = feature_selector.apply_variance_threshold(test_df, is_train=False)
        
        train_df = feature_selector.apply_correleation_filter(train_df)
        # Ensure test df keeps the same columns retained after correlation filter
        
        retrained_cols = [c for c in train_df.columns if c in test_df.columns]
        test_df = test_df[retrained_cols]
        
        train_df = feature_selector.apply_statistical_scoring(train_df, target_col=target_col, is_train=True)
        test_df = feature_selector.apply_statistical_scoring(test_df, target_col=target_col, is_train=False)
        
        joblib.dump(cat_encoder, models_dir / "fitted_categorical_encoder.pkl")
        joblib.dump(num_transformer, models_dir / "fitted_numerical_transformer.pkl")
        joblib.dump(cardinality_reducer, models_dir / "fitted_cardinality_reducer.pkl")
        joblib.dump(feature_selector, models_dir / "fitted_feature_selector.pkl")
        
        # 10. Persist Final Engineered Feature Matrices
        train_output = processed_dir / "train_features.csv"
        test_output = processed_dir / "test_features.csv"\
        
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        logger.info("==================================================")
        logger.info(" STAGE 4 COMPLETED SUCCESSFULLY!")
        logger.info(f" Train Feature Matrix Saved: {train_output} ({train_df.shape[0]:,} rows, {train_df.shape[1]} cols)")
        logger.info(f" Test Feature Matrix Saved : {test_output} ({test_df.shape[0]:,} rows, {test_df.shape[1]} cols)")
        logger.info("==================================================")
        


if __name__ == "__main__":
    run_stage4_pipeline()
        