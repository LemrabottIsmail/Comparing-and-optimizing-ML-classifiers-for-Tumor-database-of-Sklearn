from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.model_selection import RandomizedSearchCV
# For Tumor database of Sklearn
data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.3)
# Designing a kNN classifier with Hyperparameter Optimization when K = ((3, 5, 7, 9, 11, 13)
neighbors = [3, 5, 7, 9, 11, 13]
highest_accuracy = 0
bestK = 0
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    CM = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    print("When K = ", neighbor)
    print("Confusion Matrix is: ", CM)
    print("accuracy is", score)
    if score > highest_accuracy:
        highest_accuracy = score
        bestK = neighbor
    print("precision is", precision_score(y_test, y_pred))
    print("recall is", recall_score(y_test, y_pred))
    print("ROC is", roc_auc_score(y_test, y_pred))
    print("-------------------------------------")
    print("-------------------------------------")

print("The highest accuracy is {} for K = {}".format(highest_accuracy, bestK))