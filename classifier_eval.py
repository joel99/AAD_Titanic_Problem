#!/usr/bin/env python

#######################
#### Aaron McDaniel####
#######################

import numpy as np
from classifier import Classifier
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog, canny
from skimage.measure import find_contours
from PIL import Image


import time
 
def main():
    img_clf = Classifier()
    start_true = time.time()
    
    #param tuning
    p1 = [0,2,3]
    p2 = [4,5,6]
    best = [p1[0], p2[0]] # list of best params

    
    # Write headers
    file_name = "classifier/results/results.csv"
    heading = "Title\nFalse Positive, False Negative, Train Time, Avg Test, F1 Test, F2 Test, F3 Test, F4 Test, F5 Test\n"
    file = open(file_name,'w')
    file.write(heading)
    file.close()

    (data_raw, data_labels) = img_clf.load_data_from_folder('./classifier/data/')

    
    # block searching for best parameters based on cross validation
    for one in p1:
        for two in p2:
                    start = time.time()
                    
                    line = "%d,%d,%s,%f," %(c, b, n, image_time) # writing parameters used in line
                    accs = [] # List of accuracies by fold
                    for fold in range(k):
                        train_data = []
                        test_data = []
                        train_labels = []
                        test_labels = []
                        #rearange data by fold
                        for i in range(len(feature_data)):
                            if i%k == fold:
                                test_data += [feature_data[i]]
                                test_labels += [data_labels[i]]
                            else:
                                train_data += [feature_data[i]]
                                train_labels += [data_labels[i]]
                    
                        # train model and test on training data
                        img_clf.train_classifier(train_data, train_labels)

                        # document train time
                        print("Trained Fold %d in %f seconds" %(fold,(time.time()-end))) 
                        train_time = time.time()-end
                        if fold == 0:
                            line += "%f," %train_time

                        # test model
                        predicted_labels = img_clf.predict_labels(test_data)
                        
                        
                        print("Testing results:")
                        print("=============================")
                        print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
                        print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
                        print("Param 1: %d\tParam 2: %d\n" %(one, two))
                        test_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
                        #should have FP and FN
                        accs += [test_accuracy] # adds to list of fold accuracies
                        

                    avg = sum(accs)/k
                    file = open(file_name, 'a')
                    line += "%f,%f,%f,%f,%f,%f\n" %(avg,accs[0],accs[1],accs[2],accs[3],accs[4])
                    file.write(line)
                    file.close()
                        
                    # document best model
                    if avg > max:
                        max = avg
                        best = [c, b, n]
                    #print("\nCell: %d\tBlock: %d\tNorm: %s" %(c,b,n))
                    print("Avg Accuracy: %f" %avg)
                    print("Best Avg Accuracy: %f" %max)
                    print("Current Duration: %f seconds" %(time.time()-start_true))
                    print("*********************************************")
                    
        #save model with best avg accuracy
        print("Saving best model")
        print("PARAMS:\nCells: %d\tBlocks: %d\tNorm: %s" %(best[0],best[1],best[2]))

        print("Best: %s" %str(best))
        all_data = img_clf.extract_image_features(data_raw, best[0], best[1], best[2])
        print("all_labels: %d\tall_data: %d"%(len(data_labels),len(all_data)))
        img_clf.train_classifier(all_data,data_labels)
        img_clf.save("classifier/models/best.pkl")
        f = open("classifier/models/best.txt", 'w')
        f.write("Best configuration\nC: %d\tB: %d\tN: %s" %(best[0],best[1],best[2]))
        f.close()
        print("Successfully Saved Model")
            
    print("\nTotal Time: %f seconds" %(time.time()-start_true))
    
if __name__ == "__main__":
    main()
