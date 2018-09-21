#!/usr/bin/env python
"""
AAD - Titanic Dataset Paretodominance Demo
Data Parser Driver
== Team 4 ==
Aaron McDaniel
Jeffrey Minowa
Joshua Reno
Joel Ye
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
    
data_dir = 'data/'
train_fn = 'train.csv'
test_fn = 'test.csv'
test_y_fn = 'gender_submission.csv'
folds = 5

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df.values

# Returns: clean train_data, test_data
def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)
    test_data = load_data(test_fn)
    test_labels = load_data(test_label_fn)

    # Note test data has different data order
    # Convert sex column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])
    test_data[:, 3] = le.transform(train_data[:, 3])
    # Convert embark column (col 11)
    le.fit(["S", "C", "Q"])
    train_data[:, 11] = le.transform(train_data[:, 11])
    test_data[:, 10] = le.transform(train_data[:, 10])
    
    # Feature selection:
    # Trim passenger_id (c0), name (c3), ticket number (c8), cabin number (c10)
    # As we're unsure about cabin_number domain effect, we're just dropping it
    train_data = np.delete(train_data, [0, 3, 8, 10], axis = 1)
    test_data = np.delete(train_data, [0, 2, 7, 9], axis = 1)
    
    # Drop NaN rows - we have to drop corresponding rows test_data, test_labels
    keep_mask = np.copy(~pd.isnull(train_data).any(axis=1))
    train_data = train_data[keep_mask]
    x_test = test_data[keep_mask]
    y_test = test_labels[keep_mask] 

    # Separate train_data into x and y
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    return ((x_train, y_train), (x_test, y_test))

# ((train_x, train_y), (test_x, test_y))
