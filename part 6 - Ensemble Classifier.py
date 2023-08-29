import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.3)

# Optimizing the Random Forest Classifier hyperparameters ..
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
params_rf = {'n_estimators': [50, 75, 100, 125, 150, 175, 200]}
rf_gs = GridSearchCV(rf_clf, params_rf, cv=5)
rf_gs.fit(x_train, y_train)
rf_final = rf_gs.best_estimator_

# Optimizing the KNN Classifier hyperparameters ..

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': [3, 5, 7, 9, 11, 13]}
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(x_train, y_train)
knn_final = knn_gs.best_estimator_

# Training the SVM Classifier ..
ada_clf = AdaBoostClassifier()
ada_clf.fit(x_train, y_train)

print('Random Forest Score: ', accuracy_score(y_test, rf_final.predict(x_test)))
print('KNN Score: ', accuracy_score(y_test, knn_final.predict(x_test)))
print('AdaBoost Score: ', accuracy_score(y_test, ada_clf.predict(x_test)))

estimators=[('Random Forest', rf_final), ('KNN', knn_final), ('AdaBoost', ada_clf)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(x_train, y_train)
print("Ensemble Score: ", ensemble.score(x_test, y_test))