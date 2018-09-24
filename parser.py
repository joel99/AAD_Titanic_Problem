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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

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
    test_data = np.delete(test_data, [0, 2, 7, 9, 10], axis = 1)

    # Fill in NaN
    train_data = np.where(pd.isnull(train_data), -1, train_data)
    # test_data = np.where(pd.isnull(test_data), -1, test_data)
    x_test = np.where(pd.isnull(test_data), -1, test_data)
    y_test = test_labels

    # Separate train_data into x and y
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    return ((x_train, y_train), (x_test, y_test))


def score(clf, data, labels):
    """
    calculates the precision and recall for the given classifier on the given set of data and labels

    :param clf: untrained classifier to be evaluated
    :param data: the dataset used for cross validation
    :param labels: the correct labels that match with the given data
    :return: a tuple of the precision and recall scores for the given classifier
    """
    kf = KFold(shuffle=False, random_state=0)
    precision = cross_val_score(clf, data, labels, scoring='precision', cv=kf, n_jobs=-1)

    recall = cross_val_score(clf, data, labels, scoring='recall', cv=5, n_jobs=-1)
    precision = precision.mean()
    recall = recall.mean()

    return (precision, recall)

def pareto_dominance_max(ind1, ind2):
    """
    returns true if ind1 dominates ind2 by the metrics that should be maximized

    :param ind1: tuple of precision and recall scores
    :param ind2: tuple of precision and recall scores
    :return: boolean representing if ind1 dominates ind2 using metrics that should be maximized
    """

    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 < value_2:
            return False
        elif value_1 > value_2:
            not_equal = True
    return not_equal

def pareto_dominance_min(ind1, ind2):
    """
    returns true if ind1 dominates ind2 by the metrics that should be minimized

    :param ind1: tuple of FP and FN
    :param ind2: tuple of FP and FN
    :return: boolean representing if ind1 dominates ind2 using the metrics that should be minimized
    """
    not_equal = False
    for value_1, value_2 in zip(ind1, ind2):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True    
    return not_equal

def update_front(front, ind, comp):
    """
    Makes a new pareto front out of the old pareto front and new individual
    In this context an individual consists of scores and their hyper parameters
    For example ind[0] is a tuple of precision and recall scores
    and ind[1] is a list of the hyper-parameters needed to recreate the classifier

    :param front: the old pareto front to be updated
    :param ind: the new individual that may or may not change the old pareto front
    :param comp: the method used to compare individuals as being pareto dominant or not
    :return: the new pareto front
    """

    # A member belongs on the front if it dominates or is not dominated by new ind
    # New ind belongs on front if it is not dominated by any
    # If new ind dominated, rest of front won't be dominated
    newFront = []
    isNewDominated = False
    for i in range(len(front)):
        old = front[i]
        if comp(old[0], ind[0]): # Careful to compare the scores
            isNewDominated = True
            break
        if not comp(ind[0], old[0]):
            newFront.append(old)
    if isNewDominated:
        newFront.extend(front[i:]) # add rest of old front
    else:
        newFront.append(ind)
    return newFront

def convert_to_FP_FN(labels, precision, recall):
    """
    converts form precision and recall to FP and FN.
    Since Recall = TP/(TP + FN), TP = Recall * Positives
    This means we can solve for FN & FP with
    FN = TP/Recall - TP
    FP = TP/Precision - TP

    :param labels: the list of numeric labels that the precision and recall metrics came from
    :param precision: the precision of some classifier on the given labels
    :param recall: the recall of some classifier on the given labels
    :return: a tuple containing FP and FN in that order
    """
    positives = sum([1 for l in labels if l == 1])
    tp = int(recall * positives)
    fn = int(tp / recall) - tp
    fp = int(tp / precision) - tp
    return (fp, fn)
