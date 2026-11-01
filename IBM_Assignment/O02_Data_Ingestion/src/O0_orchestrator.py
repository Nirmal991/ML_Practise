import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from hygiene.O04_schema_enforcer import SchemaEnforcer
from optimization.O05_memory_optimizer import MemoryOptimizer

from O06_missingness_handler import MissingnessHandler
from O07_structural_cleaner import StructuralCleaner
from O08_outlier_detector import OutlierDetector
from O09_temporal_aligner import TemporalAligner
from O10_numerical_sanitizer import NumericalSanitizer
from O11_eda_profiler import EDAProfiler

from O12_data_leakage_validator import DataLeakageValidator
from O13_domain_feature_builder import DomainFeatureBuilder
from O14_cardinality_dimensionality_reducer import CardinalityDimensionalityReducer
from O15_categorical_encoder import CategoricalEncoder
from O16_numerical_transformer import NumericalTransformer
from O17_feature_selector import FeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage2Orchestrator")

def run_stage2_pipeline():
    logger.info("==================================================")
    logger.info(" STARTING STAGE 2: INGESTION & HYGIENE PIPELINE")
    logger.info("==================================================")
    
    # STEP 1: FILE HANDLING
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    raw_dir = project_root / "data/raw"
    qurantine_dir = project_root / "data/qurantine"
    contract_path = project_root / "config/O01_schema_contract.yaml"
    config_path = project_root / "config/O02_cleaning_rules.yaml"
    config_path4 = project_root / "config/O03_feature_rules.yaml"
    
    model_dir = project_root / "models"
    processed_dir = project_root / "data/processed"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    qurantine_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = raw_dir / "Telco-Customer-Churn.csv"
    if not target_file.exists():
        logger.warning("Dataset not found. Terminating pipeline.")
        return
    
    # STEP 1: CHECKSUM AND INTEGRITY VALIDATION
    logger.info("\n--- STEP A: File Integrity & Zero-Byte Check ---")
    is_valid_file = validate_file_integrity(target_file)
    if not is_valid_file:
        logger.critical("Pipeline halted: File failed integrity checks.")
        return
    
    # STEP 2: BATCH EXTRACTOR
    logger.info("\n--- STEP B: Landing Zone Batch Extraction ---")
    extractor = BatchFileExtractor(raw_data_file=raw_dir)
    raw_df = extractor.extract_file("Telco-Customer-Churn.csv", file_format="csv")
    
    # STEP 3: SCHEMA CONTRACT ENFORCMENT
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine Routing ---")
    enforce = SchemaEnforcer(contract_path=contract_path, qurantine_dir=qurantine_dir)
    clean_df = enforce.validate_and_qurantine(raw_df,batch_id="batch_telemetry_001")
    
    # Step 4: Low lwvwl MemoryOptimization (DownCasting)
    logger.info("\n--- STEP D: Memory Optimization & Type Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)
    
    logger.info("==================================================")
    logger.info(" STAGE 2 COMPLETED SUCCESSFULLY!")
    logger.info(f" Final Clean Records Ready for Stage 3: {len(optimized_df):,}")
    logger.info("==================================================")
    
    print("\n[FINAL CLEAN & OPTIMIZED DATAFRAME PREVIEW]")
    print(optimized_df.head())
    # print("Project root:", project_root)
    # print("First path:", project_root.parent / "data/raw/Telco-Customer-Churn.csv")
    # print("Second path:", project_root / "data/raw/Telco-Customer-Churn.csv")
    input_file = project_root / "data/raw/Telco-Customer-Churn.csv"
    if not input_file.exists():
        logger.info("file not found. Terminating the Code...")
        return
    
    logger.info(f"Initial raw dataset shape: {optimized_df.shape[0]:,} rows, {optimized_df.shape[1]} columns")
    
    #Step 5 Instantiate cleaning:
    missingness_handler = MissingnessHandler(config_path=config_path)
    structural_cleaner = StructuralCleaner(config_path=config_path)
    outlier_detector = OutlierDetector(config_path=config_path)
    temporal_aligner = TemporalAligner(config_path=config_path)
    numerical_sanitizer = NumericalSanitizer(config_path=config_path)
    eda_profiler = EDAProfiler()
    
    # STep 6 of of the Stage 3:
    logger.info("\n--- STEP 1: Missingness Analysis & Treatment ---")
    missingness_handler.analyze_missigness_report(optimized_df)
    optimized_df = missingness_handler.treat_missingness(optimized_df)
    
    logger.info("\n--- STEP 2: Structural Cleaning & Text Normalization ---")
    optimized_df = structural_cleaner.remove_duplicates(optimized_df, subset_cols=["customerID"] if "customerID" in optimized_df.columns else None)
    optimized_df = structural_cleaner.normalize_text(optimized_df)
    optimized_df = structural_cleaner.apply_caanonical_mapping(optimized_df)
    
    logger.info("\n--- STEP 3: Outlier & Anomaly Detection ---")
    optimized_df = outlier_detector.detect_and_treat_zscore(optimized_df)
#     print("\n[DEBUG] DATAFRAME COLUMN TYPES")

#     print(optimized_df.dtypes)

#     print("\n[DEBUG] NON-NUMERICAL COLUMNS")

#     print(
#     optimized_df.select_dtypes(
#         exclude=[np.number]
#     ).columns.tolist()
# )
    optimized_df = outlier_detector.detect_and_treat_iqr(optimized_df)
    optimized_df = outlier_detector.detect_multivariant_isolation_forest(optimized_df)
    
    logger.info("\n--- STEP 4: Numerical Sanitization (IEEE 754 & Infinities) ---")
    optimized_df = numerical_sanitizer.sanitize_infinities_and_nans(optimized_df)
    
    logger.info("\n--- STEP 5: Automated EDA Profiling ---")
    univariate_summary = eda_profiler.generate_univariate_summary(optimized_df)
    corr_matrix = eda_profiler.compute_correlation_matrix(optimized_df)
    target_interaction = eda_profiler.analyze_target_interactions(optimized_df, target_col="Churn")
    
    logger.info("\n[EDA PROFILING: Univariate Summary]")
    print(univariate_summary)
    logger.info("\n[EDA PROFILING: Correlation Matrix]")
    print(corr_matrix)
    logger.info ("\n[EDA PROFILING: Target Interaction]")
    print(target_interaction)
    
    # Step 7:
    output_file = processed_dir / "clean_IBM_dataset.csv"
    optimized_df.to_csv(output_file, index=False)
    
    logger.info("==================================================")
    logger.info(" STAGE 3 COMPLETED SUCCESSFULLY!")
    logger.info(f" Clean Processed Dataset Saved To: {output_file}")
    logger.info(f" Final Clean Shape: {optimized_df.shape[0]:,} rows, {optimized_df.shape[1]} columns")
    logger.info("==================================================")
    
    input_file4 = project_root / "data/processed/clean_IBM_dataset.csv"
    if input_file4.exists():
        logger.info(f"Loading the csv file {input_file4} as a dataset")
        df_cleaned = pd.read_csv(input_file4)
    else:
        logger.critical(f"[CRITICAL ERROR] No file found with {input_file4} name")
        return
    
    target_col = "Churn"
    if target_col not in df_cleaned.columns:
        raise ValueError(f"Target column '{target_col}' missing from input dataset.")

    # Remove rows without ground-truth labels and exclude the customer identifier.
    df_cleaned = df_cleaned.dropna(subset=[target_col]).copy()
    df_cleaned.drop(columns=["customerID"], errors="ignore", inplace=True)
    
    
    # stage 4 starts step 1 : Strict Train/Test Split (Preventing Global Data Leakage)
    leakage_validator = DataLeakageValidator(test_size=0.2, random_state=42)
    train_df, test_df = leakage_validator.stratified_train_test_split(df_cleaned, target_col)
    
    # 4.2. Instantiate Feature Engineering Engines
    domain_builder = DomainFeatureBuilder(config_path=config_path4)
    cardinality_reducer = CardinalityDimensionalityReducer(config_path=config_path4)
    cat_encoder = CategoricalEncoder(config_path=config_path4)
    num_transformer = NumericalTransformer(config_path=config_path4)
    feature_selector = FeatureSelector(config_path=config_path4)
    
    #4.3 Domain Feature Building (Applied independently to train and test)
    logger.info("\n--- STEP 2: Domain Feature Construction ---")
    train_df = domain_builder.build_features(train_df)
    test_df = domain_builder.build_features(test_df)
    
    # 4.4 Cardinality Reduction (Fit on train, transform test)
    cat_cols_to_reduce = [
    "PaymentMethod",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
    ]
    
    train_df = cardinality_reducer.group_rare_categories(train_df, categorical_cols=cat_cols_to_reduce,is_train=True)
    test_df = cardinality_reducer.group_rare_categories(test_df, categorical_cols=cat_cols_to_reduce,is_train=False)
    
    #4.5 Categorical Encoding (Fit on train, transform test)
    train_df = cat_encoder.fit_transform_low_cardinality(train_df, is_train=True)
    test_df = cat_encoder.fit_transform_low_cardinality(test_df, is_train=False)
    
    #4.6 Align predictor columns only; never impute or modify target labels.
    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col].copy()
    X_test = test_df.drop(columns=[target_col]).copy()
    y_test = test_df[target_col].copy()

    X_train, X_test = X_train.align(
        X_test, join="left", axis=1, fill_value=0.0
    )

    X_train.drop(columns=["customerID"], errors="ignore", inplace=True)
    X_test.drop(columns=["customerID"], errors="ignore", inplace=True)

    # Replace infinities and impute remaining numeric NaNs using TRAIN medians.
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_feature_cols = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if numeric_feature_cols:
        train_medians = X_train[numeric_feature_cols].median().fillna(0)
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].fillna(train_medians)
        X_test[numeric_feature_cols] = X_test[numeric_feature_cols].fillna(train_medians)

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    #4.7 Numerical power transformation and scaling (fit on train, transform test).
    train_df = num_transformer.fit_transform(
        train_df, is_train=True, exclude_cols=[target_col, "SeniorCitizen"]
    )
    test_df = num_transformer.fit_transform(
        test_df, is_train=False
    )

    #4.8 Multistage Feature Selection (fit on train, filter test).
    train_df = feature_selector.apply_variance_threshold(
        train_df, is_train=True, target_col=target_col
    )
    test_df = feature_selector.apply_variance_threshold(
        test_df, is_train=False, target_col=target_col
    )

    train_df = feature_selector.apply_correlation_filter(
        train_df, is_train=True, target_col=target_col
    )
    test_df = feature_selector.apply_correlation_filter(
        test_df, is_train=False, target_col=target_col
    )

    train_df = feature_selector.apply_statistical_scoring(
        train_df, target_col=target_col, is_train=True
    )
    test_df = feature_selector.apply_statistical_scoring(
        test_df, target_col=target_col, is_train=False
    )

    #4.8 Artifact Serialization (Saving fitted transformers for production inference)
    joblib.dump(cat_encoder, model_dir / "fitted_categorical_encoder.pkl")
    joblib.dump(num_transformer, model_dir / "fitted_numerical_transformer.pkl")
    joblib.dump(cardinality_reducer, model_dir / "fitted_cardinality_reducer.pkl")
    joblib.dump(feature_selector, model_dir / "fitted_feature_selector.pkl")
    
    train_output = processed_dir / "train_features.csv"
    test_output = processed_dir / "test_features.csv"
      
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    logger.info("==================================================")
    logger.info(" STAGE 4 COMPLETED SUCCESSFULLY!")
    logger.info(f" Train Feature Matrix Saved: {train_output} ({train_df.shape[0]:,} rows, {train_df.shape[1]} cols)")
    logger.info(f" Test Feature Matrix Saved : {test_output} ({test_df.shape[0]:,} rows, {test_df.shape[1]} cols)")
    logger.info("==================================================")
        
if __name__ == "__main__":
    run_stage2_pipeline()
    
    
    
    