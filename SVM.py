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
    clf = svm.SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=True, random_state=500, shrinking=True,
                  tol=0.001, verbose=False)
    #take first 200 sample
    results =clf.fit(X_test[0:1000], Y_test[0:1000])
    score=results.score(X_test, Y_test)
    
    #prediction 
    pred=results.predict(X_test)
    print(score)
    print(pred)
if __name__ == '__main__':
    main()
