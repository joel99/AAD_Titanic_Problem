#!/usr/bin/env python

#######################
#### Aaron McDaniel####
#######################

import numpy as np
from img_classifier import ImageClassifier
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog, canny
from skimage.measure import find_contours
from PIL import Image


import time

def print_pic(img):
    #for 3D or 2D grey image
    #if type(img[0][0]) is np.array or type(img[0][0]) is np.ndarray or type(img[0][0]) is list: 
    #    img = Image.fromarray(img, "RGB")
    #else:
    img = Image.fromarray(img)
    img.show()
 
def main():
    img_clf = ImageClassifier()
    start_true = time.time()
    
    #param tuning
    max = 0.0
    k = 5
    cells = [51]
    blocks = [2]
    display = False
    norms = ['L2']
    best = [cells[0], blocks[0], norms[0]]

    
    file_name = "classifier/results/results.csv"
    heading = "Hog Results of Param Search and Cross Validation Scores\ncells,blocks,norm,Image Time, Train Time, Avg Test, F1 Test, F2 Test, F3 Test, F4 Test, F5 Test\n"
    file = open(file_name,'w')
    file.write(heading)
    file.close()

    (data_raw, data_labels) = img_clf.load_data_from_folder('./classifier/data/')

    
    if not display: # block searching for best parameters based on cross validation
        for n in norms:
            for b in blocks:
                for c in cells:
                    start = time.time()
                    # automate parameter tuning
                    # load images

                    print("\n*********************************************\nProcessing Images")
                    print("Cells: %d\tBlocks: %d\tNorm: %s\n" %(c, b, n))
                    # convert images into features
                    feature_data = img_clf.extract_image_features(data_raw, c, b, n)
                    end = time.time()
                    image_time = end - start
                    print("Finished Loading images in %f seconds" %(end-start))

                    # test model

                    line = "%d,%d,%s,%f," %(c, b, n, image_time)
                    # implement heap data
                    # convert images into features
                    #print("TRD: %s\tTED: %s\tTRL: %s\tTEL: %s" %(type(train_data),type(test_data),type(train_labels),type(test_labels)))
                    accs = []
                    test = 0
                    for fold in range(k):
                        train_data = []
                        test_data = []
                        train_labels = []
                        test_labels = []
                        #rearange data
                        for i in range(len(feature_data)):
                            if i%k == fold:
                                test_data += [feature_data[i]]
                                test_labels += [data_labels[i]]
                            else:
                                train_data += [feature_data[i]]
                                train_labels += [data_labels[i]]
                    
                        # train model and test on training data
                        try:
                            img_clf.train_classifier(train_data, train_labels)
                        except:
                            train_data = [[0] for img in train_data]
                            img_clf.train_classifier(train_data, train_labels)

                        print("Trained Fold %d in %f seconds" %(fold,(time.time()-end)))
                        train_time = time.time()-end
                        if fold == 0:
                            line += "%f," %train_time

                        # test model
                        try:
                            predicted_labels = img_clf.predict_labels(test_data)
                        except:
                            test_data = [[0] for d in test_data]
                            predicted_labels = img_clf.predict_labels(test_data)
                        #print("Testing results:")
                        #print("=============================")
                        #print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
                        print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
                        print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
                        print("Cells: %d\tBlocks: %d\tNorm: %s\n" %(c, b, n))
                        print("Train Length: %d\tTest Length: %d" %(len(train_data), len(test_data)))
                        test_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
                        accs += [test_accuracy]
                        

                    avg = sum(accs)/k
                    file = open(file_name, 'a')
                    line += "%f,%f,%f,%f,%f,%f\n" %(avg,accs[0],accs[1],accs[2],accs[3],accs[4])
                    file.write(line)
                    file.close()
                        
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

    else:
        for i in [1,5]:#,10,15,20,25,30,34,35]:
            img = np.array([[value[0] for value in row] for row in test_raw[i]])
            # show orig
            #print_pic(img)
            # show sobel image
            print_pic(img)
            print_pic(center_pic(img))
            # show centered image
            #print_pic(center_pic(test_raw[i]))
  
            
    print("\nTotal Time: %f seconds" %(time.time()-start_true))
    
if __name__ == "__main__":
    main()
