from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data'][: , 3:] # petal width
y = (iris['target'] == 2).astype(np.int) # 1 if Iris Virgnica, else 0
logistic = LogisticRegression()
logistic.fit(X,y)

X_new  = np.linspace(0,3,1000).reshape(-1,1)
# 1000 values from 0 to 3, reshaped as a (-1,1) -1 makes numpy find a certain dimension to reshape it as.
# 1 just means we have one column of values

y_proba = logistic.predict_proba(X_new) # Predict the probability of
print(y_proba)
plt.plot(X_new,y_proba[:, 1], "g-", label = "Iris Virginica") # % likelihood it is iris virgnica
plt.plot(X_new,y_proba[:, 0], "b--", label = "Not Iris Virgnica") # % likelihood it is not an iris virgnica
plt.ylabel("Probability")
plt.xlabel("Petal Width")
plt.show()