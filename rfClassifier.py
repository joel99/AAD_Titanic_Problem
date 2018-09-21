"""
AAD Titanic Dataset Paretodominance Demo
Random Forest Model
- Joel Ye
"""
from sklearn.ensemble import RandomForestClassifier
from parser import load_split_all

def random_forest_test():
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    x_train, x_test, y_train, y_test = load_split_all()
    #for train_index, test_index in indices:
     #   X_train, X_test = X[train_index], X[test_index]
      #  y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

random_forest_test()
