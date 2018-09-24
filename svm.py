from classifier import Classifier
import parser
from matplotlib import pyplot as plt
from sklearn import svm

def find_best_SVM(data, labels):
    clf = svm.SVC()
    (fig, ax, x, y) = parser.init_graph()

    # param tuning SVM specific 192 - 5 = 187 combinations
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    probabilities = [True, False]
    gammas = [0.1, 0.25, 0.5, 1, 2, 5]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    folds = 5

    front = [[(10**10,10*10),(kernels[0], probabilities[0], tols[0], gammas[0])]]  # list of best scores & params

    # block searching for best parameters based on cross validation
    for k in kernels:
        gam_irrelevant = k == 'linear'
        for p in probabilities:
            for t in tols:
                for g in gammas:
                    #create and score classifier with given hyperparameters
                    clf = svm.SVC(kernel=k, probability=p, gamma=g, tol=t)
                    score = parser.score(clf, data, labels)

                    # keep track of paretofront
                    ind = [score, (k,p,t,g)]
                    front = parser.update_front(front, ind, parser.pareto_dominance_pre_rec)

                    # document performance
                    print("Params\nk: %s\tp: %s\tt: %f\tg: %f" % (k, p, t, g))
                    print("Score: %f" % (score))
                    print("*********************************************")

                    # update graph
                    parser.update_graph(fig, ax, front, x, y)

                    # only change gamma if it is relavent (SVM specific)
                    if gam_irrelevant:
                        break

        # stops graph from closing until manually closed
        plt.waitforbuttonpress()

        # return pareto front classifiers
        return generate_SVM_front(front)

def generate_SVM_front(front):
    # implements svms for each point on the pareto front
    # returns a list of SVMs
    models = []
    for ind in front:
        clf = svm.SVC(kernel=ind[1][0], probability=ind[1][1], tol=ind[1][2], gamma=ind[1][3])
        models += [clf]
    return models
