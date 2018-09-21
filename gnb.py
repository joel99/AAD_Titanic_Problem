# import numpy as np
# import re
from sklearn.naive_bayes import GaussianNB
# from skimage import io, feature, filters, exposure, color
from parser import load_split_all
from sklearn.model_selection import cross_val_score

def gaussianNB():
    x, y = load_split_all()
    return cross_val_score(GaussianNB(), x, y)

gaussianNB()