"""
AAD Titanic Dataset Paretodominance Demo
Random Forest Model
- Joel Ye
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score

def random_forest_test():
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    x_train, x_test, y_train, y_test = load_split_all()
    #for train_index, test_index in indices:
     #   X_train, X_test = X[train_index], X[test_index]
      #  y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

def find_best_rf(data, labels):
    # RF Hyperparams
    n_trees = range(5, 20)
    max_depth = range(3, 7)

    best = [n_trees[0], max_depth[0]]  # list of best params
    hyperparams = (n_trees, max_depth)

    maxScore = -1
    for h1 in hyperparams[0]:
        for h2 in hyperparams[1]:
            clf = RandomForestClassifier(n_estimators=h1, max_depth=h2)
            precision = cross_val_score(clf, data, labels, scoring=make_scorer(precision_score)).mean()
            recall = cross_val_score(clf, data, labels, scoring=make_scorer(recall_score)).mean()

            # TODO: track pareto front
            # Let's say I'm going for max precision
            score = precision
            # score = (recall + precision) / 2

            # keep track of best hyperparameters
            if score > maxScore:
                maxScore = score
                best = [h1, h2]
                
            # Document performance
            print("Params\nn_trees: %d\tmax_depth: %d" % (h1, h2))
            print("Precision: %f\tRecall: %f" % (precision, recall))
            print("*********************************************")
    print("Best Score: %f" % maxScore)
    print("Best Params: Trees=%d, Depth=%d" % (best[0], best[1]))
    return RandomForestClassifier(n_estimators=best[0], max_depth=best[1])
