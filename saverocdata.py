"""
@Time    : 2022/10/21 19:47
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: saverocdata.py
@Software: PyCharm
"""
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_auc(y_test, y_test_scores, name):
    '''

    :param y_test: test data
    :param y_test_scores: prob
    :param name: conbined name
    :return: figure
    '''

    fpr, tpr, threshod = roc_curve(y_test, y_test_scores)
    fpr = fpr.reshape(fpr.shape[0], 1)
    tpr = tpr.reshape(tpr.shape[0], 1)
    data_save = np.concatenate((fpr, tpr),axis=1)
    writerCSV = pd.DataFrame(data=data_save)

    writerCSV.to_csv('./data/%s_losses.csv'%name,encoding='utf-8')
    roc_auc = auc(fpr, tpr)

    return roc_auc

def test_xgboost(trainValue,labels,testValue,y_test):
    clf =XGBClassifier()
    name = 'xgboost'
    clf.fit(trainValue,labels)
    # predicttrainlabel = clf.predict(trainValue)
    label = clf.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name, auc)

def test_predictknn(trainValue,labels,testValue,y_test):
    name = 'knn'
    neigh =KNeighborsClassifier(n_neighbors=3)

    neigh.fit(trainValue,labels)
    label = neigh.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name, auc)

def test_predictDecisionTree(trainValue,labels,testValue,y_test):

    clf = DecisionTreeClassifier()
    name = 'DecisionTree'
    clf.fit(trainValue, labels)
    label = clf.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name, auc)

def test_predictGaussianNB(trainValue,labels,testValue,y_test):
    clf = GaussianNB(priors=None)
    clf.fit(trainValue, labels)
    name = 'GaussianNB'
    label = clf.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label,name)
    print(name, auc)

def test_predictRandomForest(trains,labels,tests,y_test):

    clf = RandomForestClassifier(random_state=0)
    clf.fit(trains, labels)
    #train
    name = 'RandomForest'
    label = clf.predict_proba(tests)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name,auc)

def test_predictSVM(trainValue,labels,testValue,y_test):

    clf = svm.SVC(gamma=0.001,probability = True)
    clf.fit(trainValue, labels)
    name = 'SVM'
    label = clf.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name,auc)

def encoder_minmax_values(df):
    dfencode = pd.get_dummies(df)
    columns = dfencode.columns
    print(columns)
    scaler = MinMaxScaler()
    normal = scaler.fit_transform(dfencode)
    data = DataFrame(normal,columns=columns)

    return data

def test_predictLogistic(trainValue,labels,testValue,y_test):

    logistic=LogisticRegression(penalty='l2',C=1,solver='lbfgs',max_iter=1000)
    logistic.fit(trainValue,labels)
    name = 'logistic'
    label = logistic.predict_proba(testValue)
    auc = plot_roc_auc(y_test,label[:,1],name)
    print(name,auc)

if __name__ == '__main__':
    df = pd.read_csv('./data/newdata4.csv')
    df = df.drop('name', axis=1)
    newdf = encoder_minmax_values(df)
    label = newdf['label']
    newdf = newdf.drop('label', axis=1)
    X = newdf
    y = label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    function_list = [test_predictLogistic, test_predictSVM, test_predictRandomForest,test_predictDecisionTree,test_predictknn,test_xgboost]
    for f in function_list:
        f(X_train, y_train, X_test,y_test)

    # label = test_predictLogistic(X_train, y_train, X_test)
    # print(classification_report(y_test, label))