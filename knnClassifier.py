from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from parser import load_split_all

def find_best_knn():
    x_train, X_test, y_train, Y_test = load_split_all()    
    cv_scores = []
    neighbors = list(range(1,50))
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]
    #determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    return KNeighborsClassifier(n_neighbors=optimal_k)
find_best_knn()