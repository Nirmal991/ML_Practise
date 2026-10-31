import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import Image, display
from io import StringIO
import pydotplus

input_file = './assets/PastHires.csv'
df = pd.read_csv(input_file)
print(df.head())


columns_names = ["feature_name", "unique_values", "dtype"]
df1 = pd.DataFrame([{"feature_name": col, "unique_values": df[col].nunique(), "dtype": df[col].dtype.name } for col in df.columns])
print(df1)
print("=" * 60)

d = {'Y':1,'N':0,'BS':0, 'MS':1, 'PhD':2 }
df.replace(d, inplace=True)

for col in df.select_dtypes(include=np.object_).columns.to_list():
    df[col] = df[col].astype(int)
    
features = list(df.columns[:6])
target = df.columns[6]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df[target]

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print("Decision Tree Classifier trained successfully.")
print("=" * 60)

y_pred = clf.predict(X_train)

accuracy = (y_pred == y_train).mean()
print("Accuracy on the training data:")
print(accuracy)

print("=" * 60)

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("tree.png")
# display(Image(graph.create_png()))
