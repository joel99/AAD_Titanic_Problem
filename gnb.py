import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from skimage import io, feature, filters, exposure, color
from parser import load_split_all
from sklearn.model_selection import cross_val_score

def gaussianNB():
    classifier = GaussianNB()
    x, y = load_split_all()
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    #for train_index, test_index in indices:
     #   X_train, X_test = X[train_index], X[test_index]
      #  y_train, y_test = y[train_index], y[test_index]
    # classifier.fit(x_train, y_train)
    # print(classifier.score(x_test, y_test))
    # print(x_train.__class__)
    # x = np.concatenate((x_train, x_test), axis = 0)
    # print(x.shape)
    # y = np.concatenate((y_train, y_test), axis = 0)
    # print(y.shape)
    print(cross_val_score(classifier, x, y))


gaussianNB()