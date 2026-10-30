# Stage 3: Data Preprocessing, Cleaning & EDA

## Learning Objectives 

1. **Diagnose Missingness Mechanisms:**
    Distinguish between MCAR, MAR, and MNAR,preserve missingness signals via indicator flags,and apply advanced multivariate KNN imputation.
2. **Execute Structural & Text Hygiene:**
    Normalize text casing,  strip whitespaces, remove duplicates, and resolve categorical typos(using Levenshtein distance fuzzy string matching (`rapidfuzz`)) 
3. **Treat Statistical & Multivariate Outliers:**
    Apply univariate bounds  (Z-Score and Tukey's IQR fence), density-based anomaly detection (Isolation Forest). 
4. **Manage Temporal Edge Cases:**
    Convert datetimes strictly to UTC, resample time-series data without introducing look-ahead bias.
5. **Sanitize Numerical Matrices:**
    Mask IEEE 754 floating-point infinities, NaNs, and prevent division-by-zero errors.
6. **Automate EDA Profiling:**
    Generate univariate statistics, bivariate correlation matrices, and feature-target interaction metrics prior to modeling.




