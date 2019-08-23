
# coding: utf-8
import sys
import csv
import random
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score



def SVM_linear():
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=10, n_jobs=-1)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    return accuracy_score(y_test, predict), np.average(scores)
    
    
def SVM_rbf():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],
                         'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=10, n_jobs=-1)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    return accuracy_score(y_test, predict), np.average(scores)

    
def Random_Forest_Regressor():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    forest = RandomForestRegressor()
    scores = cross_val_score(forest, X_train_std, y_train, cv=10)
    forest.fit(X_train_std, y_train)
    return forest.score(X_test_std, y_test), np.average(scores)
    
    
def Random_Forest_Classifier():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    forest = RandomForestClassifier()
    scores = cross_val_score(forest, X_train_std, y_train, cv=10)
    forest.fit(X_train_std, y_train)
    return forest.score(X_test_std, y_test), np.average(scores)
      
    
def DecisionTree_Classifier():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    Tree = DecisionTreeClassifier()
    scores = cross_val_score(Tree, X_train_std, y_train, cv=10)
    Tree.fit(X_train_std, y_train)
    predict = Tree.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)
  

def AdaBoost_Classifier():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ada = AdaBoostClassifier()
    scores = cross_val_score(ada, X_train_std, y_train, cv=10)
    ada.fit(X_train_std, y_train)
    predict = ada.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)
  

def Gaussian_NB():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    Gauss = GaussianNB()
    scores = cross_val_score(Gauss, X_train_std, y_train, cv=10)
    Gauss.fit(X_train_std, y_train)
    predict = Gauss.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)
    


def Linear_Discriminant_Analysis():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    Linear = LinearDiscriminantAnalysis()
    scores = cross_val_score(Linear, X_train_std, y_train, cv=10)
    Linear.fit(X_train_std, y_train)
    predict = Linear.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)
    

def Quadratic_Discriminant_Analysis():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    Quadratic = QuadraticDiscriminantAnalysis()
    scores = cross_val_score(Quadratic, X_train_std, y_train, cv=10)
    Quadratic.fit(X_train_std, y_train)
    predict = Quadratic.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)


def k_NN():
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    knn = KNeighborsClassifier(5)
    scores = cross_val_score(knn, X_train_std, y_train, cv=10)
    knn.fit(X_train_std, y_train)
    predict = knn.predict(X_test_std)
    return accuracy_score(y_test, predict), np.average(scores)
    
    
#  kadai2 dataset
train = pd.read_csv('https://raw.githubusercontent.com/Amilabinc/pattern2017-dataset/master/kadai2_train.csv', header=None)
X_train = np.hstack((train.iloc[: , :2].values, train.iloc[: , 3:].values))
y_train = train.iloc[:,0].values


#### evaluation with test data  
# kadai2 dataset
test = pd.read_csv('https://raw.githubusercontent.com/Amilabinc/pattern2017-dataset/master/kadai2_test.csv', header=None)
X_test = np.hstack((test.iloc[: , :2].values, test.iloc[: , 3:].values))
y_test = test.iloc[:,0].values

clf_all=[
    SVM_linear(),
    SVM_rbf(),
    Random_Forest_Classifier(),
    DecisionTree_Classifier(),
    AdaBoost_Classifier(),
    Gaussian_NB(),
    Linear_Discriminant_Analysis(),
    Quadratic_Discriminant_Analysis(),
    k_NN()
]

acc = [0 for i in range(9)]
out = [0 for i in range(9)]

ma=0
for i in range(9):
    out[i],acc[i] = clf_all[i]

ma=max(acc)
for i in range(9):
    if acc[i] == ma: 
        print('answer acc : %s' % out[i])


print('acc of train')
print(acc)
print('acc of test')
print(out)

