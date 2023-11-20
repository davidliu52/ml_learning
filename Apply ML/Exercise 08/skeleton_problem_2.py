# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:08:55 2023

@author: merte
"""

"""
Problem 2: fitting a data set
"""
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import KFold
import numpy as np

def accuracy(y_gt, y_pred):
    """ Returns the accuracy score."""

    return np.sum(y_gt == y_pred) / len(y_gt)

# load data from disk
data = np.loadtxt('dataset_exercise_08.txt', delimiter=',')
X = data[:, :2]  # inputs
y = data[:, -1]  # targets


# create 5-fold cross validation data splitting
kf = KFold(n_splits=5)

X_train, X_val = [], []  # <-- lists of 5 numpy arrays
y_train, y_val = [], []
for train_index, val_index in kf.split(X):
    X_train.append(X[train_index, :])
    X_val.append(X[val_index, :])

    y_train.append(y[train_index])
    y_val.append(y[val_index])


def cross_validate(X_train, y_train, X_val, y_val, predictor):
    # return bias and variance of a predictor fitted to the training data
    # and evaluated on the validation data (here: k-fold splits)
    # predictor MUST have a .fit() and .predict() method
    acc = []
    for idx, _ in enumerate(X_train):
        # fit predictor and evaluate
        predictor.fit(X_train[idx], y_train[idx])
        y_pred = predictor.predict(X_val[idx])
        
        # store accuracy value
        acc.append(accuracy(y_val[idx], y_pred))

    return np.mean(acc), np.std(acc)


"""
Trying out different models
"""

scores = dict()

# try a DT (default parameters)
DTClassifier = DT()

acc_DT_mean, acc_DT_std = cross_validate(
    X_train, y_train, X_val, y_val, DTClassifier)

scores['DT_default'] = [acc_DT_mean, acc_DT_std]
