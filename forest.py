import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score

def main():
    
    dataset=pd.read_csv("dataUpdated.csv")
    dataset.dropna(how='all', inplace = True)
    X = dataset.drop(labels = ["publisher", "developer", "name", "sale"], axis = 1)
    Y = dataset["sale"]
    
    
    X = X.reset_index(drop=True)
	
    print(X_test)
    clf = RandomForestClassifier()
	
    results =clf.fit(X_train, Y_train) 
    
    #prediction 
    pred=results.predict(X_test)
	
	#accuracy score
    result = accuracy_score(Y_test, pred, normalize=True)
    print(result)
if __name__ == '__main__':
    main()
