import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# predict Exam success
# Features: hours_studied, has_pencil, minutes_studied, breakfast_eaten
# Target: passed_exam

# 1. hours_studied --> useful! Driectly impacts exam success
# 2. has_pencil --> (0 variance) not useful for prediction (type of feature : quasi-constant)
# 3. minutes_studied ---> redundant (correlated with hours_studied) (100% correlation)
# if minutes_studied is considered , it would be overfitting the model, and introduce multicollinearity, so it should be dropped.
# Does it introduce VIF (Variance Inflation Factor) --> Yes, because it is perfectly correlated with hours_studied
# 4. breakfast_eaten ---> Binary Feature (0 = not eaten or 1 = eaten).
# should it be dropped? --> Not necessarily, it could be useful for prediction depending on the context
# 5. passed_exam --> Target variable (Y), should not be used as a feature for prediction 

data = {
    "hours_studied": [1, 2, 5, 6, 8, 9, 10, 12, 14, 15],
    "has_pencil": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 
    "minutes_studied": [60, 120, 300, 360, 480, 540, 600, 720, 840, 900],
    "breakfast_eaten": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    "passed_exam": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)
print(df)
print("-"*60)
#STEP 1 Seperate predictors and Target:
# X --> Predictors (Features)
# y --> Target (Label)

X = df[["hours_studied", "has_pencil", "minutes_studied", "breakfast_eaten"]]
y = df["passed_exam"]

print("Original Feature (X):")
print(X)
print("-"*60)

# STEP 2: Filter method Remove Quasi Constant Features

variance_selector = VarianceThreshold(threshold=0.1275)
X_variance_filtered = variance_selector.fit_transform(X)
print("Features after Variance Threshold Filtering:")
print(X_variance_filtered)
print("_" * 60)
surviving_features = X.columns[variance_selector.get_support()]
# surviving_features = variance_selector.get_support()
print("Surviving Features after Variance Threshold Filtering:")
print(surviving_features)
dropped_features = X.columns[~variance_selector.get_support()]
print("Dropped Features after Variance Threshold Filtering:")
print(dropped_features)
print("-"*60)

# Rebuild DataFrame
X_step1_df = pd.DataFrame(X_variance_filtered, columns=surviving_features)
print("Rebuilt DataFrame with Surviving Features:")
print(X_step1_df)

# Step 3 : Remove correlated features (multicollinearity) : Removing redundancy
corr_matrix = X_step1_df.corr().abs()
print("Correlation Matrix:")
print(corr_matrix)
print("_" * 60)
threshold = 0.85
feature_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i,j] > threshold:
            feature_to_drop.add(corr_matrix.columns[i])
            
print("Features to drop due to high Correlation")# Drop the redundant columns due to high correlation
print(feature_to_drop)
print("-"*60)

# Drop the redundant columns due to high correlation
X_final = X_step1_df.drop(columns=feature_to_drop)
print("Final Features after Removing Highly Correlated Features:")
print(X_final)
print("_" * 60)

# Step 4
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42,stratify=y)

print("Training Features (X_train):")
print(X_train)
print("Testing Features (X_test):")
print(X_test)
print("Training Labels (y_train):")
print(y_train)
print("Testing Labels (y_test):")
print(y_test)
print("_" * 60)

# why Logistic Regression? 
# Logistic Regression is a simple and interpretable model for binary classification tasks.
# It works well when the relationship between the features and the target is approximately linear in the log-odds space.
model = LogisticRegression()
# Fit the model on the training data
model.fit(X_train, y_train)

#Evaluate on testing data
accuracy = model.score(X_test, y_test)
print("Model Accuracy on Test Data:", accuracy)

# Predict the labels for the test set
y_pred = model.predict(X_test)
print("Predicted Labels for Test Data:")
print(y_pred)
print("Actual Labels for Test Data:")
print(y_test.values)
print("_" * 60)

# Perfomance matrix (Confusion Matrix)
print(f"Accuracy Score : {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))