import parser
from matplotlib import pyplot as plt
from sklearn import svm

def find_best_SVM(data, labels):
    clf = svm.SVC()
    (fig, sc) = parser.init_graph()

    # param tuning SVM specific 192 - 5 = 187 combinations
    #kernels = ['linear','poly', 'rbf', 'sigmoid']
    kernels = ['rbf']
    probabilities = [False, True]
    #probabilities=[False]
    tols = [0.00001, 0.0001, 0.001, 0.01]
    #tols = [0.001]
    folds = 5

    front = []  # list of best scores & params

    print("starting svm search")

    # block searching for best parameters based on cross validation
    for k in kernels:
        for p in probabilities:
            for t in tols:
                print("Params\nk: %s\tp: %s\tt: %f" % (k, p, t))

                #create and score classifier with given hyperparameters
                clf = svm.SVC(kernel=k, probability=p, tol=t)
                score = parser.score(clf, data, labels)
                score = parser.convert_to_FP_FN(labels, score[0], score[1])

                # keep track of paretofront
                ind = [score, (k,p,t)]
                front = parser.update_front(front, ind, parser.pareto_dominance_min)

                # document performance
                print("FP: %f\tFN: %f" % (score[0],score[1]))
                print("*********************************************")

                parser.update_graph(fig, sc, front)

    ''' # stops graph from closing until manually closed
    try:
        plt.waitforbuttonpress()
    except:
        print("Graph Closed")'''

        # return pareto front classifiers
    return generate_SVM_front(front)

def generate_SVM_front(front):
    # implements svms for each point on the pareto front
    # returns a list of SVMs
    models = []
    for ind in front:
        clf = svm.SVC(kernel=ind[1][0], probability=ind[1][1], tol=ind[1][2])
        models += [clf]
    return models
