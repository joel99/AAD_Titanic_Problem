from sklearn.naive_bayes import GaussianNB
from parser import load_split_all
from sklearn.model_selection import cross_val_score

def gaussianNB():
    classifier = GaussianNB()
    x, y = load_split_all()
    print(cross_val_score(classifier, x, y))
    return classifier