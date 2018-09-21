#!/usr/bin/env python

#######################
#### Aaron McDaniel####
#######################

import numpy as np
import pandas as pd
from classifier import Classifier
from sklearn import svm, metrics


import time

path = './data/'


def main():
    clf = Classifier()

    #param tuning
    kernals = ['linear','poly','rbf','sigmoid']
    probabilities =[True, False]
    gammas =[0.1,0.25,0.5,1,2,5]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    folds = 5
    
    best = [kernals[0], probabilities[0], tols[0], gammas[0]] # list of best params
    (data, labels) = read_data()

    # block searching for best parameters based on cross validation
    for k in kernals:
        gam_irrelevent = k == 'linear'
        for p in probabilities:
            for t in tols:
                for g in gammas:
                    start = time.time()
                    
                    fps = [] # List of false positives by fold
                    fns = [] # List of false negatives by fold
                    for fold in range(folds):
                        (train_data, train_labels, test_data, test_labels) = make_fold(data,labels,folds,fold)
                    
                        # train model and test on training data
                        clf.create_SVM(k,p,t,g)
                        clf.train(train_data, train_labels)

                        # test model
                        predicted_labels = clf.predict(test_data)

                        # calculate FP and FN
                        total = len(test_data)
                        fp = sum([1 for i in range(total) if predicted_labels[i] == 1 and test_labels[i] == 0])
                        fn = sum([1 for i in range(total) if predicted_labels[i] == 0 and test_labels[i] == 1])
                        fps += [fp]
                        fns += [fn]

                    avgFP = sum(fps)/folds
                    avgFN = sum(fns)/folds
                    score = (avgFP + avgFN) / 2

                    # document best model
                    if score > max:
                        max = score
                        best = [k,p,t,g]
                    print("Params\nk: %s\tp: %s\tt: %f\tg: %f" %(k,p,t,g))
                    print("Avg FP: %f\tAVG FN: %f\ttotal: %d" %(avgFP,avgFN,len(data)))
                    print("Best Avg Score: %f" %max)
                    print("*********************************************")
                    
                    #only change gamma if it is relavent (SVM specific)
                    if gam_irrelevent:
                        break
                    
        #return best model
        return clf.create_SVM(best[0],best[1],best[2],best[3])


if __name__ == "__main__":
    main()

def read_data():
    # turns the data into a list of data and lables

    train_labels = pd.read_csv(path + "train.csv",usecols=[1]).values
    test_labels = pd.read_csv(path + "gender_submission.csv",usecols=[1]).values
    train_data = pd.read_csv(path + "train.csv",usecols=[0,2,3,4,5,6,7,8,9,10,11]).values
    test_data = pd.read_csv(path + "test.csv").values

    data = np.array(list(train_data) + list(test_data))
    labels = np.array(list(train_labels) + list(test_labels))
    labels = np.array([l[0] for l in labels])

    data = np.delete(data, [0, 2, 7, 9, 10], axis=1)


    #does not handle strings
    return (data, labels)

def filter_strings(data):
    # returns the data with the strings changed
    # replaces None with -1
    # replace "male" and "female" with 0 and 1 respectively
    # replaces all letters with their location in the alphabet (i.e. 'a' -> 0, 'ab' -> 01)
    for line in data:
        for i in range(len(data)):
            #res

def make_fold(data, labels, total, sel):
    # turns data and labels into train and test sets
    # total -> total number of folds (fraction of data used for test)
    # sel -> which portion of data will be test data
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    # rearange data by fold
    for i in range(len(data)):
        if i % total == sel:
            test_data += [data[i]]
            test_labels += [labels[i]]
        else:
            train_data += [data[i]]
            train_labels += [labels[i]]

    return (train_data, train_labels, test_data, test_labels)


def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)

    # Note test data has different data order
    # Convert gender column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])

    # Feature selection, trim passenger_id (col 0), label (col 1) and name (col 3), embark location, cabin number, ticket number
    train_data = np.delete(train_data, [0, 3, 8, 10, 11], axis=1)

    # Drop NaN for now. See imputing
    train_data = train_data[~pd.isnull(train_data).any(axis=1)]
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    xtr, xte, ytr, yte = train_test_split(x_train, y_train)
    # kf = KFold(n_splits=folds)
    # indices = kf.split(train_data, test_data)
    return xtr, xte, ytr, yte  # Ideally would like an interface to request organized data

