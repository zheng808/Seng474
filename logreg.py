from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



X = pd.read_csv('dataUpdated.csv', encoding = "utf8")
print(X.shape)
X.dropna(how='all', inplace = True)
#X = X[np.notnull(X)]

print(X.head())

#one hot encoding. Memory consumption too large (creates about 19k columns for developers and publishers)
#X = pd.get_dummies(X, columns = ["publisher", "developer"])

#TODO: use publisher and developer in data
Y = X["sale"]
X = X.drop(labels = ["publisher", "developer", "name", "sale"], axis = 1)

X = X.reset_index(drop=True)

#TODO: add validation set for tuning hyperparameters
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, Y_train)

print (logisticRegr.score(X_test, Y_test))


