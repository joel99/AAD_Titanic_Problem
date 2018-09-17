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

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

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

    def extract_image_features(self, data, cell=51, block=2, norm='L2'):
        # Please do not modify the header above

        # extract feature vector from image data
        feature_data = []
        for img in data:
            #hog implementation -> 97.5% cell:(12,12) block:(1,1) L1-sqrt TOO EASY
            if type(img[0][0]) == list or type(img[0][0]) == np.array or type(img[0][0]) == np.ndarray:
                img = np.array([[value[0] for value in row] for row in img])
            else:
                img = np.array([[value for value in row] for row in img])
            f = feature.hog(img, orientations=10, pixels_per_cell=(cell, cell), cells_per_block=(block, block), feature_vector=True, block_norm=norm)
            feature_data.append(f)
            #feature_data += feature.hog(img, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(4, 4), feature_vector=True, block_norm='L2-Hys')
            # has 100% accuracy
            #feature_data += [hog(img, pixels_per_cell=(cell,cell),cells_per_block=(block,block),block_norm=norm)]
		#
        # Please do not modify the return type below
        return(feature_data)

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
        