"""
AAD Titanic Dataset Paretodominance Demo
Random Forest Model
- Joel Ye
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score
import parser

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
    (fig, sc) = parser.init_graph()


    front = []
    hyperparams = (n_trees, max_depth)

    maxScore = -1
    for h1 in hyperparams[0]:
        for h2 in hyperparams[1]:
            clf = RandomForestClassifier(n_estimators=h1, max_depth=h2)
            score = parser.score(clf, data, labels)
            score = parser.convert_to_FP_FN(labels, score[0], score[1])

            ind = [score, (h1, h2)]
            front = parser.update_front(front, ind, parser.pareto_dominance_min)

            parser.update_graph(fig, sc, front)

            # Document performance
            print("Params\nn_trees: %d\tmax_depth: %d" % (h1, h2))
            print("Precision: %f\tRecall: %f" % (score[0], score[1]))
            print("*********************************************")
    return generate_RF_front(front)

def generate_RF_front(front):
    # implements svms for each point on the pareto front
    # returns a list of SVMs
    models = []
    for ind in front:
        clf = RandomForestClassifier(n_estimators=ind[1][0], max_depth=ind[1][1])
        models += [clf]
    return models

