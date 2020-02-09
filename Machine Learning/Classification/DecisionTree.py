from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, 2:] #petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth = 2) #Distance from the root to the leaf, we will try to find the optimal
tree_clf.fit(X,y)

preds = tree_clf.predict_proba([[5,1.5]]) # petal length and width
#predict_proba gives the probability that it is in each section. In this case there is Iris Setosa, Iris Versicolor, and
# Iris Virgnica. It gives the likelihood of the flower being in each category.
print(preds)

preds2 = tree_clf.predict([[5,1.5]]) #Outputs the category with highest percentage. In this case it is Iris Versicolor
print(preds2)


#Testing the optimal max depth
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
max_depth = [1,2,3,4,5,6]
accuracy = []
for i in max_depth:
    clf = DecisionTreeClassifier(max_depth = i)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    accuracy.append(score)
print("The highest accuracy is with a depth of:", max_depth[accuracy.index(max(accuracy))] , " and an accuracy of",max(accuracy))

plt.plot(max_depth,accuracy)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()