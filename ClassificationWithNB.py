import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes is a method of classification
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

x = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)
# Note that Random_state = 42 is used because we want reproducible results

gnb = GaussianNB()
gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print(preds) #For this particular example, 1 means malignant and 0 means bening for cancer.

#Accuracy
print(accuracy_score(y_test, preds))