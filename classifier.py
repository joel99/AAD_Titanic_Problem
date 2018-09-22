#!/usr/bin/env python

#######################
#### Aaron McDaniel ###
#### Robert Adams   ### 
#######################

from sklearn import svm, metrics

class Classifier:

    def __init__(self):
        self.classifier = None

    # model for the train_classifier methods, each classifier should get their own methods
    # take in classifier params as input variables
    def train_classifier_SVM(self, train_data, train_labels, kernal, probability, tol, gamma):
        # Overwrite the classifier and train it to be an SVM with the given params
        # params: 
        # kernal(string)
        # gamma(float)
        # probability (bool)
        # tol (float)
        
        # train model and save the trained model to self.classifier
        self.classifier = svm.SVC(kernel=kernal, gamma=gamma, probability=probability, tol=tol)
        self.classifier.fit(train_data, train_labels)

        return None

    def predict_labels(self, data):
        # Please do not modify the header
        predicted_labels = self.classifier.predict(data)
        # Please do not modify the return type below
        return predicted_labels
