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
folds = 5

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df.values

def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)

    # Note test data has different data order
    # Convert gender column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])
    
    # Feature selection, trim passenger_id (col 0), label (col 1) and name (col 3), embark location, cabin number, ticket number
    train_data = np.delete(train_data, [0, 3, 8, 10, 11], axis = 1)

    # Drop NaN for now. See imputing
    train_data = train_data[~pd.isnull(train_data).any(axis=1)]
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    xtr, xte, ytr, yte = train_test_split(x_train, y_train)
    # kf = KFold(n_splits=folds)
    # indices = kf.split(train_data, test_data)
    # return xtr, xte, ytr, yte # Ideally would like an interface to request organized data
    return np.concatenate((xtr, xte), axis = 0), np.concatenate((ytr, yte), axis = 0)
