import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier, plot_tree
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target
print(X.head())
print(y[:5])

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_split=4, min_samples_leaf=2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\n Confusion matrix")
print(confusion_matrix(y_test, y_pred))

print(classification_report(
    y_test, y_pred, target_names=iris.target_names # specifies the names of the classes (labels) that you want to display in the classification report instead of numerical labels.
))

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Classifier")

plt.show()