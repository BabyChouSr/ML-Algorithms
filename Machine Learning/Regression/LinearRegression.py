from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.datasets import load_boston

#Boston is just an example of a csv; you can use any
boston = load_boston()

#This is the beginning of the training of the model / Fitting of the model
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=1)
regr = LinearRegression() # Linear Regression for numbers / can also be logistic regression
regr.fit(X_train, y_train)

#Gives a prediction according to the X value
y_pred = regr.predict(X_test)
print(r2_score(y_test, y_pred))
#In statistics also known as the r^2 which is the proportion of variability accounted for by the model
#Good r^2 should be about 0.75+

#Prints out the error
print(median_absolute_error(y_test, y_pred)) # The difference between the true value and predicted
