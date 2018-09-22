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

def find_best_SVM(data, labels):
    clf = Classifier()
    (fig, ax, x, y) = init_graph()

    # param tuning SVM specific 192 - 5 = 187 combinations
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    probabilities = [True, False]
    gammas = [0.1, 0.25, 0.5, 1, 2, 5]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    folds = 5

    front = [[(10**10,10*10),(kernels[0], probabilities[0], tols[0], gammas[0])]]  # list of best scores & params

    # block searching for best parameters based on cross validation
    for k in kernels:
        gam_irrelevant = k == 'linear'
        for p in probabilities:
            for t in tols:
                for g in gammas:
                    #create and score classifier with given hyperparameters
                    clf.create_SVM(k, p, t, g)
                    score = score(clf.classifier, data, labels)

                    # keep track of paretofront
                    ind = [score, (k,p,t,g)]
                    front = update_front(front, ind)

                    # document performance
                    print("Params\nk: %s\tp: %s\tt: %f\tg: %f" % (k, p, t, g))
                    print("Score: %f" % (score))
                    print("*********************************************")

                    # update graph
                    update_graph(fig, ax, front, x, y)

                    # only change gamma if it is relavent (SVM specific)
                    if gam_irrelevant:
                        break

        # return pareto front classifiers
        return generate_SVM_front(clf, front)

def score(clf, data, labels):
    precision = cross_val_score(clf.classifier, data, labels, scoring=precision_score)
    recall = cross_val_score(clf.classifier, data, labels, scoring=recall_score)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)

    return (precision, recall)

# returns true if ind1 dominates ind2
def pareto_dominance(ind1, ind2):
    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True
    return not_equal

def update_front(front, ind):
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

def generate_SVM_front(clf, front):
    models = []
    for ind in front:
        clf.make_SVM(ind[1][0], ind[1][1], ind[1][2], ind[1][3])
        models += [clf.classifier]
    return models

def init_graph():
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
    x.clear()
    y.clear()
    x.append([ind[0][0] for ind in front])
    y.append([ind[0][1] for ind in front])
    sc.set_offsets(np.c_[x, y])
    fig.canvas.draw_idle()
    plt.pause(0.1)

def driver():
    train, test = load_split_all()
    train_x, train_y = train
    test_x, test_y = test
    best_rf = find_best_rf(train_x, train_y)
    svm_front = find_best_SVM(train_x, train_y)
    plt.waitforbuttonpress()

driver()
