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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from classifier import Classifier
from rfClassifier import find_best_rf
data_dir = 'data/'
train_fn = 'train.csv'
test_fn = 'test.csv'
test_label_fn = 'gender_submission.csv'
folds = 5

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df.values

# Returns: clean train_data, test_data
def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)
    test_data = load_data(test_fn)
    test_labels = load_data(test_label_fn)

    # Note test data has different data order
    # Convert sex column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])
    test_data[:, 3] = le.transform(test_data[:, 3])
    # Convert embark column (col 11)
    # le.fit(["S", "C", "Q", None])
    # print(train_data[:, 11])
    # train_data[:, 11] = le.transform(train_data[:, 11])
    # test_data[:, 10] = le.transform(test_data[:, 10])
    
    # Feature selection:
    # Trim passenger_id (c0), name (c3), ticket number (c8), cabin number (c10)
    # As we're unsure about cabin_number domain effect, we're just dropping it
    # Dropping embark since we think it's not too helpful, and has NaN
    train_data = np.delete(train_data, [0, 3, 8, 10, 11], axis = 1)
    test_data = np.delete(train_data, [0, 2, 7, 9, 10], axis = 1)
    
    # Fill in NaN
    train_data = np.where(pd.isnull(train_data), -1, train_data)
    x_test = np.where(pd.isnull(test_data), -1, test_data)
    y_test = test_labels

    # Separate train_data into x and y
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    return ((x_train, y_train), (x_test, y_test))

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
        clf.create_SVM(best[0], best[1], best[2], best[3])
        return clf.classifier

def driver():
    train, test = load_split_all()
    train_x, train_y = train
    test_x, test_y = test
    best_rf = find_best_rf(train_x, train_y)
driver()
