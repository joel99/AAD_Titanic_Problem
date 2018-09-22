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
    def make_SVM(self, kernal, probability, tol, gamma):
        # Overwrite the classifier and make it an SVM with the given params
        # params: 
        # kernal(string)
        # gamma(float)
        # probability (bool)
        # tol (float)
        
        # train model and save the trained model to self.classifier
        self.classifier = svm.SVC(kernel=kernal, gamma=gamma, probability=probability, tol=tol)

        return None

    def train(self, data, labels):
        # trains the classifier by calling the fit function
        self.classifier.fit(data, labels)

    def predict(self, data):
        # returns a list of labels for the inputted data
        predicted_labels = self.classifier.predict(data)
        return predicted_labels
