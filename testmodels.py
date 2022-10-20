"""
@Time    : 2022/10/18 15:39
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: testmodels.py
@Software: PyCharm
"""
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
import pandas as pd


def test_predictknn(trainValue,labels,testValue):
    neigh =KNeighborsClassifier(n_neighbors=3)

    neigh.fit(trainValue,labels)
    predicttrainlabel = neigh.predict(trainValue)
    label = neigh.predict(testValue)
    return label,predicttrainlabel

def test_predictDecisionTree(trainValue,labels,testValue):

    clf = DecisionTreeClassifier()
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

def test_predictGaussianNB(trainValue,labels,testValue):

    clf = GaussianNB(priors=None)
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

def test_predictRandomForest(trains,labels,tests):

    clf = RandomForestClassifier(random_state=0)
    clf.fit(trains, labels)
    #train
    predicttrainlabel = clf.predict(trains)
    label = clf.predict(tests)
    return label,predicttrainlabel

def test_predictSVM(trainValue,labels,testValue):

    clf = svm.SVC(gamma=0.001)
    clf.fit(trainValue, labels)
    label = clf.predict(testValue)
    return label

def encoder_minmax_values(df):
    dfencode = pd.get_dummies(df)
    columns = dfencode.columns
    print(columns)
    scaler = MinMaxScaler()
    normal = scaler.fit_transform(dfencode)
    data = DataFrame(normal,columns=columns)

    return data

def test_predictLogistic(trainValue,labels,testValue):

    logistic=LogisticRegression(penalty='l2',C=1,solver='lbfgs',max_iter=1000)
    logistic.fit(trainValue,labels)
    label = logistic.predict(testValue)

    return label

if __name__ == '__main__':
    df = pd.read_csv('./data/newdata4.csv')
    df = df.drop('name', axis=1)
    newdf = encoder_minmax_values(df)
    label = newdf['label']
    newdf = newdf.drop('label', axis=1)
    X = newdf
    y = label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    label = test_predictLogistic(X_train, y_train, X_test)
    print(classification_report(y_test, label))