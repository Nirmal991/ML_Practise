import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# DATASET: Predicting Job Hiring (20 records)
# ==========================================
# Designed to intentionally showcase:
# 1. Quasi-constant feature (laptop_owned -> almost all 1s)
# 2. Redundant / highly correlated feature (coding_score & typing_speed)
# 3. Missing values (interview_score has NaNs)
# 4. Outlier (years_experience has an extreme value: 35)
# 5. Categorical with rare categories (preferred_os has 'Linux' only once)
# 6. Skewed distribution (salary_expectation)
# ==========================================

data = {
    "years_experience": [1, 2, 3, 2, 4, 5, 6, 3, 5, 7, 8, 2, 4, 6, 5, 3, 35, 4, 5, 2],  #--->Outliers
    "coding_score":     [45, 50, 60, 55, 65, 70, 75, 58, 72, 80, 85, 52, 68, 78, 74, 59, 95, 66, 71, 48],
    "typing_speed":     [46, 51, 61, 56, 66, 71, 76, 59, 73, 81, 86, 53, 69, 79, 75, 60, 96, 67, 72, 49], # --> Highly correlated Feature
    "laptop_owned":     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],# --> Quasi feature
    "interview_score":  [70, 65, np.nan, 80, 85, 90, 75, 60, np.nan, 95, 88, 72, 78, 82, 88, 65, 92, 70, 80, 62], #-->Missing values 
    "preferred_os":     ["Windows", "Mac", "Windows", "Linux", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", 
                         "Windows", "Mac", "Windows", "Mac", "Windows", "Windows", "Mac", "Windows", "Mac", "Windows"], #--> Categorical rare categories
    "salary_expectation": [30000, 32000, 35000, 33000, 40000, 45000, 50000, 34000, 46000, 55000, 
                           60000, 32000, 41000, 52000, 48000, 35000, 150000, 42000, 47000, 31000], # --> Outlier
    "hired":            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0] # --> target variable
}

df = pd.DataFrame(data)
print(f"Original Dataset Shape: {df.shape}")
print("-" * 60)
X = df.drop(columns=["hired"])
y = df["hired"]
print("All columns list: ", X.columns.tolist())

# 1. Zero 
vt_zero = VarianceThreshold(threshold=0)
X_num = X.select_dtypes(include=[np.number])
# print("Numeric columns before the zero variance: ", X_num.columns)
vt_zero.fit(X_num)
X_num = X_num.loc[:,vt_zero.get_support()]
# print("Numeric columns After the zero variance: ", len(X_num.columns))

# 2 variance threshold quasi 
quasi_threshold = 0.06
vt_quasi = VarianceThreshold(threshold=quasi_threshold)
vt_quasi.fit(X_num)
X_num = X_num.loc[:,vt_quasi.get_support()]
# print("Numeric columns After the Quasi variance removal: ", X_num.columns)

# 3 Co variance Threshold
# print(X_num.corr().round(4))
corr_threshold = 0.996
corr_matrix = X_num.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
X_num = X_num.drop(columns=to_drop)
print("Numeric Columns After Highly Correlated Features Removed: ", X_num.columns.tolist())

# Missing Values Imputation
print(X_num.isna().sum())
X_num["inteview_score_missing"] = X_num["interview_score"].isna().astype(int)
X_num["interview_score"] = X_num["interview_score"].fillna(X_num["interview_score"].median())
# print(X_num)

#Outlier Treatment:
cap = X_num["years_experience"].quantile(0.95)
X_num["years_experience"] = np.where(X_num['years_experience'] > cap,cap,X_num["years_experience"])
# print(X_num)

X_cat = X.select_dtypes(include=['object','str']).copy()
counts = X_cat["preferred_os"].value_counts()
X_cat["preferred_os"] = X_cat["preferred_os"].replace(counts[counts < 2].index,"other")
X_cat = pd.get_dummies(X_cat, columns=["preferred_os"],drop_first=True)
# print(X_cat)



