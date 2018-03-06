from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def plotGraph(prediction, Y_test):
    plt.scatter(Y_test, prediction)  
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()


X = pd.read_csv('dataUpdated.csv', encoding = "utf8")

#print(X.shape)
X.dropna(how='all', inplace = True)

#one hot encoding. Memory consumption too large (creates about 19k columns for developers and publishers)
#X = pd.get_dummies(X, columns = ["publisher", "developer"])

#TODO: use publisher and developer in data
Y = X["sale"]
X = X.drop(labels = ["publisher", "developer", "name", "sale"], axis = 1)

X = X.reset_index(drop=True)

#TODO: add validation set for tuning hyperparameters
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=0)
print(X_test.shape)
print(Y_test.shape)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
results = logisticRegr.fit(X_train, Y_train)

X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
Y_train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)

prediction = logisticRegr.predict_proba(X_test)

#take first 20 sample
#prediction = prediction.flatten()[0:20]
#Y_test = Y_test[0:20]

print("prediction", prediction)

score = logisticRegr.score(X_test, Y_test)
print("score-----", score)










