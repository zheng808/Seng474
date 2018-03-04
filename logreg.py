from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


X = pd.read_csv('data.csv', encoding = "utf8")
print(X.shape)
X.dropna(how='all', inplace = True)

X = X.reset_index(drop=True)

print(X.head())


X = X.data[:,:,2]
print(X.head())

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()




