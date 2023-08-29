import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['class'] = data.target

classes = np.unique(data.target) # 0 and 1

classes_names = np.unique(data.target_names) # benign and malignant

classes = dict(zip(classes, classes_names)) # mapping between numbers and classes

df['class'] = df['class'].replace(classes)
# replace class columns 0,1 with benign and malignant


x = df.drop(columns="class")
y = df["class"]
feature_names = x.columns
label_names = y.unique()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# Designing a Decision Tree Classifier with Hyperparameter Optimization of Depth = (3, 4, 5, 6)

depths = [3, 4, 5, 6]
highest_accuracy = 0
hyperparameter = 0
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(x_train, y_train)
    print("When depth is ", depth)
    print(export_text(tree, feature_names=list(feature_names)))
    y_pred = tree.predict(x_test)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    score = accuracy_score(y_test, y_pred)
    if score > highest_accuracy:
        highest_accuracy = score
        hyperparameter = depth
    print("Accuracy is ", score)
    print("---------------------------------------------------")
    print("---------------------------------------------------")

print("The highest accuracy is {} for Depth = {}".format(highest_accuracy, hyperparameter))