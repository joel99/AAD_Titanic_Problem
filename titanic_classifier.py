#!/usr/bin/env python

"""
Automated Algorithm Design VIP
Titanic Dataset Paretodominance Demo
TitanicClassifier Base Class
== Team 4 ==
Aaron McDaniel
Jeffrey Minowa
Joshua Reno
Joel Ye
============
"""
from sklearn.externals import joblib

class TitanicClassifier:

    def __init__(self, classifier = None):
        self.classifier = classifier

    def train_classifier(self, train_data, train_labels):
        if self.classifier == None:
            print("Error: Classifier not initialized, expected to run set_classifier first")
            return
        self.classifier.fit(list(train_data), train_labels)

    def set_classifier(self, classifier):
        if self.classifier != None:
            print("Warning: Overwriting existing classifier")
        self.classifier = classifier
        
    def predict_labels(self, data):
        if self.classifier == None:
            print("Error: Classifier not initialized, expected to run set_classifier first")
            return
        predicted_labels = self.classifier.predict(data)
        return predicted_labels
    
    def save(self, fileName):
        joblib.dump(self.classifier, fileName)
    
    def load(self, fileName):
        self.classifier = joblib.load(fileName)
        
