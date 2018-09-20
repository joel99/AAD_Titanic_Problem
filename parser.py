#!/usr/bin/env python

"""
Automated Algorithm Design VIP
Titanic Dataset Paretodominance Demo
Data Parser Driver
== Team 4 ==
Aaron McDaniel
Jeffrey Minowa
Joshua Reno
Joel Ye
============
"""
import pandas as pd

data_dir = 'data/'

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df
