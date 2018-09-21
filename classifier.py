#!/usr/bin/env python

#######################
#### Aaron McDaniel ###
#### Robert Adams   ### 
#######################

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog
from sklearn.externals import joblib

class Classifier:

    def __init__(self):
        self.classifier = None

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        #create one large array of image data
        data = io.concatenate_images(ic)

        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data,labels)
    
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
    
    def save(self, fileName):
        joblib.dump(self.classifier, fileName)
    
    def load(self, fileName):
        self.classifier = joblib.load(fileName)
        