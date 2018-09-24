#!/usr/bin/env python

#######################
#### Aaron McDaniel####
#######################

import numpy as np
import pandas as pd
from classifier import Classifier
from sklearn import svm, metrics


import time

path = './data/'


def main():
    clf = Classifier()
    start_true = time.time()
    

    #param tuning
    kernals = ['linear','poly','rbf','sigmoid']
    probabilities =[True, False]
    gammas =[0.1,0.25,0.5,1,2,5]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    folds = 5
    
    best = [kernals[0], probabilities[0], tols[0], gammas[0]] # list of best params
    
    # Write headers
    file_name = "./results/svm_results.csv"
    heading = "SVM classifier performance\nKernal,Probability,Tol,Gamma,Avg FP,Avg FN,Total,Avg Accuracy,Train Time\n"
    file = open(file_name,'w')
    file.write(heading)
    file.close()
    
    (data, labels) = read_data()

    # block searching for best parameters based on cross validation
    for k in kernals:
        gam_irrelevent = k == 'linear'
        for p in probabilities:
            for t in tols:
                for g in gammas:
                    start = time.time()
                    
                    line = "%s,%s,%f,%f," %(k, p, t, g) # writing parameters used in line
                    accs = [] # List of accuracies by fold
                    fps = [] # List of false positives by fold
                    fns = [] # List of false negatives by fold
                    for fold in range(folds):
                        (train_data, train_labels, test_data, test_labels) = make_fold(data,labels,folds,fold)
                    
                        # train model and test on training data
                        clf.train_classifier_SVM(train_data, train_labels, k, p, t, g)

                        # document train time
                        print("Trained Fold %d in %f seconds" %(fold,(time.time()-end))) 
                        train_time = time.time()-end
                        if fold == 0:
                            line += "%f," %train_time

                        # test model
                        predicted_labels = clf.predict_labels(test_data)
                        
                        # display testing results
                        print("Testing results:")
                        print("=============================")
                        print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
                        print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))
                        print("Param 1: %d\tParam 2: %d\n" %(one, two))
                        test_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
                
                        # calculate FP and FN
                        total = len(test_data)
                        fp = sum([1 for i in range(total) if predicted_labels[i] == 1 and test_labels[i] == 0])
                        fn = sum([1 for i in range(total) if predicted_labels[i] == 0 and test_labels[i] == 1])
                        
                        accs += [test_accuracy] # adds to list of fold accuracies
                        fps += [fp]
                        fns += [fn]
                        

                    avgA = sum(accs)/folds
                    avgFP = sum(fps)/folds
                    avgFN = sum(fns)/folds
                    line += "%f,%f,%d,%f\n" %(avgFP, avgFN, int(len(data)/folds), avgA)
                    file = open(file_name, 'a')
                    file.write(line)
                    file.close()
                        
                    # document best model
                    if avg > max:
                        max = avg
                        best = [k,p,t,g]
                    #print("\nCell: %d\tBlock: %d\tNorm: %s" %(c,b,n))
                    print("Avg Accuracy: %f\tAvg FP: %f\tAVG FN: %f" %(avgA,avgFP,avgFN))
                    print("Best Avg Accuracy: %f" %max)
                    print("Current Duration: %f seconds" %(time.time()-start_true))
                    print("*********************************************")
                    
                    #only change gamma if it is relavent
                    if gam_irrelevent:
                        break
                    
        #save model with best avg accuracy
        print("Saving best model")
        print("PARAMS:\nKernal: %s\tProb: %s\tTol: %f\tGamma: %f" %(best[0],best[1],best[2],best[3]))
        (all_data, all_labels) = read_data()
        clf.train_classifier(all_data,all_labels,best[0],best[1],best[2],best[3])
        clf.save("./models/best.pkl")
        f = open("./models/best.txt", 'w')
        f.write("Best configuration\nK: %s\tP: %s\tT: %f\tG: %f" %(best[0],best[1],best[2],best[3]))
        f.close()
        print("Successfully Saved Model")
            
    print("\nTotal Time: %f seconds" %(time.time()-start_true))
    
if __name__ == "__main__":
    main()

def read_data():
    # turns the data into a list of data and lables

    train_labels = pd.read_csv(path + "train.csv",usecols=[1]).values
    test_labels = pd.read_csv(path + "gender_submission.csv",usecols=[1]).values
    train_data = pd.read_csv(path + "train.csv",usecols=[0,2,3,4,5,6,7,8,9,10,11]).values
    test_data = pd.read_csv(path + "test.csv").values

    data = np.array(list(train_data) + list(test_data))
    labels = np.array(list(train_labels) + list(test_labels))
    labels = np.array([l[0] for l in labels])

    #does not handle strings
    return (data, labels)

def filter_strings(data):
    # returns the data with the strings changed
    # replaces None with -1
    # replace "male" and "female" with 0 and 1 respectively
    # replaces all letters with their location in the alphabet (i.e. 'a' -> 0, 'ab' -> 01)
    for line in data:
        for i in range(len(data)):
            i = i


def make_fold(data, labels, total, sel):
    # turns data and labels into train and test sets
    # total -> total number of folds (fraction of data used for test)
    # sel -> which portion of data will be test data
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    # rearange data by fold
    for i in range(len(data)):
        if i % total == sel:
            test_data += [data[i]]
            test_labels += [labels[i]]
        else:
            train_data += [data[i]]
            train_labels += [labels[i]]

    return (train_data, train_labels, test_data, test_labels)


