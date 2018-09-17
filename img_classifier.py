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

class ImageClassifier:

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

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above

        # train model and save the trained model to self.classifier
        self.classifier = svm.SVC(kernel = "linear")
        self.classifier.fit(list(train_data), train_labels)

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
        