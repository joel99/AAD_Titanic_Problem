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
import matplotlib.pyplot as plt
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


def score(clf, data, labels):
    """
    calculates the precision and recall for the given classifier on the given set of data and labels

    :param clf: untrained classifier to be evaluated
    :param data: the dataset used for cross validation
    :param labels: the correct labels that match with the given data
    :return: a tuple of the precision and recall scores for the given classifier
    """

    precision = cross_val_score(clf.classifier, data, labels, scoring=precision_score)
    recall = cross_val_score(clf.classifier, data, labels, scoring=recall_score)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)

    return (precision, recall)

def pareto_dominance(ind1, ind2):
    """
    returns true if ind1 dominates ind2

    :param ind1: tuple of precision and recall scores
    :param ind2: tuple of precision and recall scores
    :return: boolean representing if ind1 dominates ind2
    """

    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True
    return not_equal

def update_front(front, ind):
    """
    Makes a new pareto front out of the old pareto front and new individual
    In this context an individual consists of scores and their hyper parameters
    For example ind[0] is a tuple of precision and recall scores
    and ind[1] is a list of the hyper-parameters needed to recreate the classifier

    :param front: the old pareto front to be updated
    :param ind: the new individual that may or may not change the old pareto front
    :return: the new pareto front
    """

    all = front + [ind]

    front = []
    for ind1 in all:
        pareto = True
        for ind2 in front:
            if pareto_dominance(ind2[0],ind1[0]):
                # ind1 cannot belong on the pareto front
                pareto = False
                break
            elif pareto_dominance(ind1[0],ind2[0]):
                #ind1 belongs on pareto front and ind2 does not
                front.remove(ind2)
        if pareto:
            front += [ind1]
    return front

def init_graph():
    """
    Creates a matplotlib.pyplot scatter graph that can be continuously updated
    as new pareto fronts are made
    ues the plt.waitforbuttonpress() method to keep the graph displayed
    after it is finished being updated

    :param Fig: the figure being displayed
    :param Sc: the scatterplot displayed on the graph
    :param X: the list of x-axis coordinates being graphed
    :param Y: the list of y-axis coordinates being graphed
    :return: a list of variables needed to update the graph.
    """
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    plt.draw()
    plt.show()
    return (fig, sc, x, y)

def update_graph(fig, sc, front, x, y):
    """
    updates the given figure to display the given pareto front

    :param fig: the figure being displayed
    :param sc: the scatter plot associated with the figure
    :param front: the list of individuals being graphed.
                  In this context an individual consists of scores and their hyper parameters.
                  For example ind[0] is a tuple of precision and recall scores
                  and ind[1] is a list of the hyper-parameters needed to recreate the classifier
    :param x: the list of x-axis coordinates associated with the graph
    :param y: the list of y-axis coordinates associated with the graph
    :return:
    """

    x.clear()
    y.clear()
    x.append([ind[0][0] for ind in front])
    y.append([ind[0][1] for ind in front])
    sc.set_offsets(np.c_[x, y])
    fig.canvas.draw_idle()
    plt.pause(0.1)