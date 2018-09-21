#!/usr/bin/env python
"""
AAD - Titanic Dataset Paretodominance Demo
Data Parser Driver
== Team 4 ==
Aaron McDaniel
Jeffrey Minowa
Joshua Reno
Joel Ye
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from classifier import Classifier
data_dir = 'data/'
train_fn = 'train.csv'
test_fn = 'test.csv'
folds = 5

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df.values

def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)

    # Note test data has different data order
    # Convert gender column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])
    
    # Feature selection, trim passenger_id (col 0), label (col 1) and name (col 3), embark location, cabin number, ticket number
    train_data = np.delete(train_data, [0, 3, 8, 10, 11], axis = 1)

    # Drop NaN for now. See imputing
    train_data = train_data[~pd.isnull(train_data).any(axis=1)]
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    xtr, xte, ytr, yte = train_test_split(x_train, y_train)
    # kf = KFold(n_splits=folds)
    # indices = kf.split(train_data, test_data)
    # return xtr, xte, ytr, yte # Ideally would like an interface to request organized data
    return np.concatenate((xtr, xte), axis = 0), np.concatenate((ytr, yte), axis = 0)


def find_best_SVM(data, labels):
    clf = Classifier()

    # param tuning SVM specific
    kernals = ['linear', 'poly', 'rbf', 'sigmoid']
    probabilities = [True, False]
    gammas = [0.1, 0.25, 0.5, 1, 2, 5]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    folds = 5

    best = [kernals[0], probabilities[0], tols[0], gammas[0]]  # list of best params

    # block searching for best parameters based on cross validation
    for k in kernals:
        gam_irrelevent = k == 'linear'
        for p in probabilities:
            for t in tols:
                for g in gammas:
                    #create and score classifier with given hyperparameters
                    clf.create_SVM(k, p, t, g)
                    precision = cross_val_score(clf.classifier, data, labels, scoring=precision_score)
                    recall = cross_val_score(clf.classifier, data, labels, scoring=recall_score)
                    precision = sum(precision)/len(precision)
                    recall = sum(recall)/len(recall)
                    score = (recall + precision) / 2

                    # keep track of best hyperparameters
                    if score > max:
                        max = score
                        best = [k, p, t, g]

                    # document performance
                    print("Params\nk: %s\tp: %s\tt: %f\tg: %f" % (k, p, t, g))
                    print("Precision: %f\tRecall: %f" % (precision, recall))
                    print("Best Avg Score: %f" % max)
                    print("*********************************************")

                    # only change gamma if it is relavent (SVM specific)
                    if gam_irrelevent:
                        break

        # return best model
        return clf.create_SVM(best[0], best[1], best[2], best[3])
