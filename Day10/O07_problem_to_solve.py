import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import Image, display


input_file = "./assets/PastHires.csv"
df = pd.read_csv(input_file)
d = {'Y':1,'N':0,'BS':0, 'MS':1, 'PhD':2 }
df.replace(d, inplace=True)
df = df.astype({col: int for col in df.select_dtypes(include=np.object_).columns.to_list()})

# scaling the features
features = list(df.columns[:6])
target = df.columns[6]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df[target]

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf = clf.fit(X_train, y_train)
print("Random Forest Classifier trained successfully.")
print("=" * 60)

# Predicting on the training data itself (for demonstration purposes)
y_pred = clf.predict(X_train)

# Evaluating the accuracy on the training data
accuracy = (y_pred == y_train).mean()
print("Accuracy on the training data:")
print(accuracy)
print("=" * 60)
sample_1 = pd.DataFrame([[10, 1, 4, 0, 0, 0]], columns=features)
sample_2 = pd.DataFrame([[10, 0, 4, 0, 0, 0]], columns=features)

print("Prediction for sample 1:")
print(clf.predict(sample_1))
print("Prediction for sample 2:")
print(clf.predict(sample_2))
print("=" * 60)