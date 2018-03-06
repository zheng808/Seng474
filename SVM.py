import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def main():
    
    dataset=pd.read_csv("dataUpdated.csv")
    dataset.dropna(how='all', inplace = True)
    X = dataset.drop(labels = ["publisher", "developer", "name", "sale", "days_since_sale", "average_days_per_sale_variance", "months_since_release"], axis = 1)
    months = X['month']
    year = X[['year']]
    Y = dataset["sale"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=0)
    
    X = X.reset_index(drop=True)
    C=1.0
    print(X_test)
    clf = svm.SVC(C=C, kernel='rbf', probability=True)
    #take first 200 sample
    results =clf.fit(X_test[0:200], Y_test[0:200]) 
    score=results.score(X_test, Y_test)
    
    #prediction 
    pred=results.predict(X_test)
    print(score)
    print(pred)
if __name__ == '__main__':
    main()