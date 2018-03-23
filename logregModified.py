from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable 
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.ensemble import BaggingClassifier


def plotGraph(prediction, Y_test):
    plt.scatter(Y_test, prediction)  
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.show()


X = pd.read_csv('dataUpdated.csv', encoding = "utf8")

#print(X.shape)
X.dropna(how='all', inplace = True)

all_zero_scores = []
model_scores = []
svm_scores = []
ensemble_scores = []

#one hot encoding. Memory consumption too large (creates about 19k columns for developers and publishers)
#X = pd.get_dummies(X, columns = ["publisher", "developer"])

rows1 = X.loc[(X['months_since_release'] < 6) & (X['year'] == 2018)]
rows = pd.merge(X.loc[(X['months_since_release'] > 4) & (X['year'] == 2018)], rows1)
names = rows['name'].tolist()#list of names to omit
X = X[~X['name'].isin(names)]#remove from dataset for training
	
#get all names of games and remove duplicates (by putting into set and converting back to list)
all_names = list(set(X['name'].tolist()))
all_names.pop(0)#index 0 is nan
iterations = 25
for id, name in enumerate(all_names):
	if (id > iterations):
		break
		
	print ("name: ", name)
	rows = X.loc[(X['name'] != name)]
	names = rows['name'].tolist()#list of names to omit
	newX = X[~X['name'].isin(names)]#remove from dataset for training

	#TODO: use publisher and developer in data
	Y = newX["sale"]
	newX = newX.drop(labels = ["publisher", "developer", "name", "sale", "days_since_sale", "average_days_per_sale", "average_days_per_sale_variance"], axis = 1)

	newX = newX.reset_index(drop=True)

	#TODO: add validation set for tuning hyperparameters
	X_train, X_test, Y_train, Y_test = train_test_split(newX, Y, test_size = 0.3)
	print(X_train.shape)

	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	Y_train=np.asarray(Y_train)
	Y_test=np.asarray(Y_test)
	
	print("\nlogistic regression")
	#logistic regression model
	logisticRegr = LogisticRegression()
	try:#bug with scikit learn that throws valueerror when all predictions are of the same class (with logistic regression).
		results = logisticRegr.fit(X_train, Y_train)
	except ValueError:
		print("skip")
		continue
	prediction = logisticRegr.predict(X_test)
	score = accuracy_score(Y_test, prediction, normalize=True)
	model_scores.append(score)
	print("score-----", score)
	print (prediction)

	#zeroes
	print ("\nguessing with all 0's")
	d = pd.DataFrame(np.zeros((len(Y_test), 1)))
	score = accuracy_score(d, Y_test, normalize=True)
	print("score----", score)
	all_zero_scores.append(score)
    
    #svm
	print ("\nsvm")
	clf = svm.SVC(C=1.0, kernel='rbf', probability=True)
    #take first 200 sample
	results =clf.fit(X_train, Y_train)
	prediction = clf.predict(X_test)
	score=accuracy_score(Y_test, prediction, normalize=True)
	print (prediction)
	print("score----", score)
	svm_scores.append(score)
	
	#ensemble types
	print ("\nensemble")
	clf = BaggingClassifier(n_estimators = 40)
	clf =clf.fit(X_train, Y_train) 
	pred=clf.predict(X_test)
	print(pred)
	result = accuracy_score(Y_test, pred, normalize=True)
	ensemble_scores.append(result)
	print("score-----", result)
	
	print("\n\n")
	
	
# plotting

N = len(all_zero_scores)  # number of groups
ind = np.arange(N)  # group positions
width = 0.2  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([model_scores[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([all_zero_scores[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')
p3 = ax.bar(ind + (width * 2), np.hstack(([svm_scores[:-1], [0]])), width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + (width * 3), np.hstack(([ensemble_scores[:-1], [0]])), width,
            color='red', edgecolor='k')

# plot annotations
ax.set_xticks(ind + (width * 2))
ax.set_xticklabels(all_names[:iterations],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Scores')
plt.legend([p1[0], p2[0], p3[0], p4[0]], ['model', 'zeroes'], loc='upper left')
plt.show()










