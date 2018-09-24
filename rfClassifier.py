"""
AAD Titanic Dataset Paretodominance Demo
Random Forest Model
- Joel Ye
"""
from sklearn.ensemble import RandomForestClassifier
import parser

def find_best_rf(data, labels):
    print("Searching for best random forest")
    # RF Hyperparams
    n_trees = range(5, 20)
    max_depth = range(3, 7)
    # (fig, sc) = parser.init_graph()
    
    front = []
    hyperparams = (n_trees, max_depth)

    for h1 in hyperparams[0]:
        for h2 in hyperparams[1]:
            clf = RandomForestClassifier(n_estimators=h1, max_depth=h2)
            score = parser.score(clf, data, labels)
            score = parser.convert_to_FP_FN(labels, score[0], score[1])

            ind = [score, (h1, h2)]
            front = parser.update_front(front, ind, parser.pareto_dominance_min)

            # parser.update_graph(fig, sc, front)

            # Document performance
            """
            print("Params\nn_trees: %d\tmax_depth: %d" % (h1, h2))
            print("Precision: %f\tRecall: %f" % (score[0], score[1]))
            print("*********************************************")
            """
    return generate_RF_front(front)

# Generate RF classifier front, given params
def generate_RF_front(front):
    return [RandomForestClassifier(n_estimators=ind[1][0], max_depth=ind[1][1]) for ind in front]

