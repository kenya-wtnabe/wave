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
from sklearn.metrics import classification_report, confusion_matrix



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')


def SVM_linear_p():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    sc = StandardScaler()
    sc.fit(X_train_p)
    X_train_std = sc.transform(X_train_p)
    X_test_std = sc.transform(X_test_p)
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train_std, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test_std)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train_std)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test_std)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    

    # Show a graph
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def SVM_rbf_p():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    sc = StandardScaler()
    sc.fit(X_train_p)
    X_train_std = sc.transform(X_train_p)
    X_test_std = sc.transform(X_test_p)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],
                         'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train_std, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test_std)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train_std)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test_std)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    

    # Show a graph
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    
def Random_Forest_Regressor_p():
    forest = RandomForestRegressor()
    forest.fit(X_train_p, y_train)
    
    ##### evaluation with training data 
    y_train_pred = forest.predict(X_train_p)
    y_test_pred = forest.predict(X_test_p)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def Random_Forest_Classifier_p():
    forest = RandomForestClassifier()
    forest.fit(X_train_p, y_train)
    
    ##### evaluation with training data 
    y_train_pred = forest.predict(X_train_p)
    y_test_pred = forest.predict(X_test_p)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    
def DecisionTree_Classifier_p():
    Tree = DecisionTreeClassifier()
    Tree.fit(X_train_p, y_train)
    
    ##### evaluation with training data 
    y_train_pred = Tree.predict(X_train_p)
    y_test_pred = Tree.predict(X_test_p)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=Tree, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def AdaBoost_Classifier_p():
    ada = AdaBoostClassifier()
    ada.fit(X_train_p, y_train)
    
    ##### evaluation with training data 
    y_train_pred = ada.predict(X_train_p)
    y_test_pred = ada.predict(X_test_p)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=ada, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def Gaussian_NB_p():
    Gauss = GaussianNB()
    Gauss.fit(X_train_p, y_train)
    
    predict = Gauss.predict(X_train_p)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Gauss.predict(X_test_p)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=Gauss, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def Linear_Discriminant_Analysis_p():
    Linear = LinearDiscriminantAnalysis()
    Linear.fit(X_train_p, y_train)
    
    predict = Linear.predict(X_train_p)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Linear.predict(X_test_p)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=Linear, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def Quadratic_Discriminant_Analysis_p():
    Quadratic = QuadraticDiscriminantAnalysis()
    Quadratic.fit(X_train_p, y_train)
    
    predict = Quadratic.predict(X_train_p)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Quadratic.predict(X_test_p)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=Quadratic, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def k_NN_p():
    knn = KNeighborsClassifier(3)
    knn.fit(X_train_p, y_train)
    
    predict = knn.predict(X_train_p)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = knn.predict(X_test_p)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    # Show a graph
    X_combined = np.vstack((X_train_p, X_test_p))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length[standard size]')
    plt.ylabel('petal width[standard size]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def SVM_linear_sc():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train_std, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test_std)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train_std)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test_std)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    
def SVM_rbf_sc():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],
                         'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train_std, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test_std)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train_std)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test_std)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))

def SVM_linear():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    
def SVM_rbf():
    ######################
    # Edit below code.
    # Let's try to rise accuracy rate!
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1],
                         'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    print (clf.best_estimator_)

    for params, mean_score, all_scores in clf.grid_scores_:
        print ("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)
    print (classification_report(y_true, y_pred))
    
    svm = clf.best_estimator_
    
    #####################

    ##### evaluation with training data 
    predict = svm.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = svm.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))

    

def Random_Forest_Regressor():
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    
    ##### evaluation with training data 
    y_train_pred = forest.predict(X_train)
    
    y_test_pred = forest.predict(X_test)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
    
    
def Random_Forest_Classifier():
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)
    
    ##### evaluation with training data 
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
      
    
def DecisionTree_Classifier():
    Tree = DecisionTreeClassifier()
    Tree.fit(X_train, y_train)
    
    ##### evaluation with training data 
    y_train_pred = Tree.predict(X_train)
    y_test_pred = Tree.predict(X_test)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
  

def AdaBoost_Classifier():
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    
    ##### evaluation with training data 
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    
    print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('r2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)) )
  

def Gaussian_NB():
    Gauss = GaussianNB()
    Gauss.fit(X_train, y_train)
    
    predict = Gauss.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Gauss.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    


def Linear_Discriminant_Analysis():
    Linear = LinearDiscriminantAnalysis()
    Linear.fit(X_train, y_train)
    
    predict = Linear.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Linear.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    


def Quadratic_Discriminant_Analysis():
    Quadratic = QuadraticDiscriminantAnalysis()
    Quadratic.fit(X_train, y_train)
    
    predict = Quadratic.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = Quadratic.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    


def k_NN():
    knn = KNeighborsClassifier(3)
    knn.fit(X_train, y_train)
    
    predict = knn.predict(X_train)
    print('train: ', accuracy_score(y_train, predict))
    print('train: ', classification_report(y_train, predict))
    
    predict = knn.predict(X_test)
    print('test: ', accuracy_score(y_test, predict))
    print('test: ', classification_report(y_test, predict))
    
    
#  kadai1 dataset 
train = pd.read_csv('https://raw.githubusercontent.com/Amilabinc/pattern2017-dataset/master/kadai1_train.csv', header=None)
X_train = train.iloc[: , 1:].values
y_train = train.iloc[:,0].values


#### evaluation with test data  
# kadai1 dataset
test = pd.read_csv('https://raw.githubusercontent.com/Amilabinc/pattern2017-dataset/master/kadai1_test.csv', header=None)
X_test = test.iloc[: , 1:].values
y_test = test.iloc[:,0].values

print(len(X_train),len(X_test))

X_train_p = train.iloc[: , [1,2]].values
X_test_p = test.iloc[: , [1,2]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

X_train_p = train.iloc[: , [1,3]].values
X_test_p = test.iloc[: , [1,3]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

X_train_p = train.iloc[: , [1,4]].values
X_test_p = test.iloc[: , [1,4]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

X_train_p = train.iloc[: , [2,3]].values
X_test_p = test.iloc[: , [2,3]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

X_train_p = train.iloc[: , [2,4]].values
X_test_p = test.iloc[: , [2,4]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

X_train_p = train.iloc[: , [3,4]].values
X_test_p = test.iloc[: , [3,4]].values
SVM_linear_p()
SVM_rbf_p()
Random_Forest_Regressor_p()
Random_Forest_Classifier_p()
DecisionTree_Classifier_p()
AdaBoost_Classifier_p()
Gaussian_NB_p()
Linear_Discriminant_Analysis_p()
Quadratic_Discriminant_Analysis_p()
k_NN_p()

SVM_linear()
SVM_rbf()
SVM_linear_sc()
SVM_rbf_sc()
Random_Forest_Regressor()
Random_Forest_Classifier()
DecisionTree_Classifier()
AdaBoost_Classifier()
Gaussian_NB()
Linear_Discriminant_Analysis()
Quadratic_Discriminant_Analysis()
k_NN()

