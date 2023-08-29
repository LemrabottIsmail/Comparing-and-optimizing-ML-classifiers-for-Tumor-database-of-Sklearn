import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# Designing a Random Forest Classifier
estimators = [50, 75, 100, 125, 150, 175, 200]
highest_accuracy = 0
hyperparameter = 0
for num in estimators:
    tree = RandomForestClassifier(n_estimators=num, random_state=44)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    print("When No of estimators is ", num)
    y_pred = tree.predict(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    score = accuracy_score(y_test, y_pred)
    if score > highest_accuracy:
        highest_accuracy = score
        hyperparameter = num
    print("Accuracy is ", score)
    print("---------------------------------------------------")
    print("---------------------------------------------------")

print("The highest accuracy is {} for number of estimators = {}".format(highest_accuracy, hyperparameter))