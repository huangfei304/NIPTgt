#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

## ML MODELS
CLASSIFIERS = [RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),    \
               AdaBoostClassifier(random_state=42, n_estimators=100),                   \
               xgb.sklearn.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1), \
               KNeighborsClassifier(),                                                  \
               LogisticRegression(random_state=42),                                     \
               VotingClassifier(estimators=[("rf", RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)), \
               ('ada',AdaBoostClassifier(random_state=42, n_estimators=100)),           \
               ('xgb', xgb.sklearn.XGBClassifier(random_state=42, n_estimators=100))], voting='soft')]

## average accuracy
def average_accuracy_score(y_test, y_pred,classes):
    all_acc = []
    r = ""
    y_pred_new = pd.Series(y_pred)
    y_pred_new.index=y_test.index
    for c in classes:
        c_acc = accuracy_score(y_test[y_test==c], y_pred_new[y_test[y_test==c].index])
        r +="{} accuracy-{}\t".format(c,round(c_acc,2))
        all_acc.append(c_acc)
    r +="avaraged accuray-{}".format(round(sum(all_acc)/len(all_acc),2))
    return r

def model_scores(train_data,test_data,clf):
    X_train = train_data.iloc[:,0:-1]
    y_train = train_data.iloc[:,-1]
    X_test = test_data.iloc[:,0:-1]
    y_test = test_data.iloc[:,-1]
    classes = sorted(set(y_train))
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    average_accuracy = average_accuracy_score(y_test, y_pred,classes)
    return accuracy,average_accuracy

def main(trainfile,predfile,outfile):
    """
    """
    results=[]
    train_data = pd.read_csv(trainfile, sep='\t')
    test_data = pd.read_csv(predfile,sep='\t')
    #train_data, test_data = train_test_split(all_data, test_size=0.2)

    for clf in CLASSIFIERS:
        clf_name = clf.__class__.__name__
        print("clf:{}".format(clf_name),file=sys.stderr)
        model_acc,model_avg_acc = model_scores(train_data,test_data,clf)
        results.append([clf_name, model_acc,model_avg_acc])

    results_df = pd.DataFrame(results,columns=['clf_name','model_acc','model_avg_acc'])
    results_df.to_csv(outfile, index=False, header=True, sep='\t')

if __name__ == '__main__':
    (trainfile,predfile,outfile)=sys.argv[1:]
    main(trainfile,predfile,outfile)
