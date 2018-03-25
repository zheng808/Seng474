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
from collections import defaultdict
import collections


dataset=pd.read_csv("dataUpdated.csv")
dataset.dropna(how='all', inplace = True)
	
rows = dataset.loc[(dataset['name'] != 'Counter-Strike: Global Offensive')]
#rows = dataset.loc[(dataset['name'] != 'Ukrainian ball in search of gas')]
names = rows['name'].tolist()#list of names to omit
dataset = dataset[~dataset['name'].isin(names)]#remove from dataset for training

sales = {}
for month in range(1, 13):
	for year in  range(2010, 2019):
		sales[str(year) + str(month)] = 0;
	
	
plt.clf()
plt.title("")
for index, set in dataset.iterrows():
	#print (str(set["year"]))
	if (set["year"] >= 2010 and set["sale"] == 1):
		sales[str(set["year"]) + str(set["month"])] += 1

sales = collections.OrderedDict(sorted(sales.items()))
plotsX = []
plotsY = []
for year in  range(2012, 2018):
	for month in range(1, 13):
		plotsX.append(year + (month / 12))
		plotsY.append(sales[str(year) + str(month)])
	
plt.ylim([0, 2])
plt.plot(plotsX, plotsY, '-o')
plt.show()