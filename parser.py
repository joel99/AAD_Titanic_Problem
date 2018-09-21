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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from classifier import Classifier
data_dir = 'data/'
train_fn = 'train.csv'
test_fn = 'test.csv'
test_y_fn = 'gender_submission.csv'
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
    test_data[:, 3] = le.transform(train_data[:, 3])
    # Convert embark column (col 11)
    le.fit(["S", "C", "Q"])
    train_data[:, 11] = le.transform(train_data[:, 11])
    test_data[:, 10] = le.transform(train_data[:, 10])
    
    # Feature selection:
    # Trim passenger_id (c0), name (c3), ticket number (c8), cabin number (c10)
    # As we're unsure about cabin_number domain effect, we're just dropping it
    train_data = np.delete(train_data, [0, 3, 8, 10], axis = 1)
    test_data = np.delete(train_data, [0, 2, 7, 9], axis = 1)
    
    # Drop NaN rows - we have to drop corresponding rows test_data, test_labels
    keep_mask = np.copy(~pd.isnull(train_data).any(axis=1))
    train_data = train_data[keep_mask]
    x_test = test_data[keep_mask]
    y_test = test_labels[keep_mask] 

    # Separate train_data into x and y
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    return ((x_train, y_train), (x_test, y_test))

def find_best_SVM(data, labels):
    clf = Classifier()

    # param tuning
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
                    start = time.time()

                    fps = []  # List of false positives by fold
                    fns = []  # List of false negatives by fold
                    for fold in range(folds):
                        (train_data, train_labels, test_data, test_labels) = make_fold(data, labels, folds, fold)

                        # train model and test on training data
                        clf.create_SVM(k, p, t, g)
                        clf.train(train_data, train_labels)

                        # test model
                        predicted_labels = clf.predict(test_data)

                        # calculate FP and FN
                        total = len(test_data)
                        fp = sum([1 for i in range(total) if predicted_labels[i] == 1 and test_labels[i] == 0])
                        fn = sum([1 for i in range(total) if predicted_labels[i] == 0 and test_labels[i] == 1])
                        fps += [fp]
                        fns += [fn]

                    avgFP = sum(fps) / folds
                    avgFN = sum(fns) / folds
                    score = (avgFP + avgFN) / 2

                    # document best model
                    if score > max:
                        max = score
                        best = [k, p, t, g]
                    print("Params\nk: %s\tp: %s\tt: %f\tg: %f" % (k, p, t, g))
                    print("Avg FP: %f\tAVG FN: %f\ttotal: %d" % (avgFP, avgFN, len(data)))
                    print("Best Avg Score: %f" % max)
                    print("*********************************************")

                    # only change gamma if it is relavent (SVM specific)
                    if gam_irrelevent:
                        break

        # return best model
        return clf.create_SVM(best[0], best[1], best[2], best[3])

def driver():
    train, test = read_split_all()
    train_x, train_y = train
    test_x, test_y = test
