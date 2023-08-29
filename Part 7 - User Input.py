from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

breastCancerData = load_breast_cancer()

X = breastCancerData.data[:, :2] # For the first two features ..
y = breastCancerData.target

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

input1 = int(input("Enter feature 1 (mean radius) : "))
input2 = int(input("Enter feature 2 (mean texture) : "))
x_test = [[input1, input2]]

classes = ["benign", "malignant"]
estimators = []

for name, clf in zip(names, classifiers):

    clf.fit(x_train, y_train)
    estimators.append((name, clf))
    pred = clf.predict(x_test)
    print("The {} classifier predicted the class of ' {} ' based on your input data".format(name, classes[pred[0]]))

ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(x_train, y_train)
pred = clf.predict(x_test)

print("The Ensemble classifier predicted the class of ' {} ' based on your input data".format(classes[pred[0]]))