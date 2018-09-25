from sklearn.neighbors import KNeighborsClassifier
import parser

def find_best_knn(data, labels):
    print("Searching for best knn")
    neighbors = list(range(1,50))
    # (fig, sc) = parser.init_graph()
    front = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = parser.score(knn, data, labels)
        precision, recall = scores
        score = parser.convert_to_FP_FN(labels, precision, recall)
        individual = [score, [k]]
        front = parser.update_front(front, individual, parser.pareto_dominance_min)
        # parser.update_graph(fig, sc, front)

    # return generate_KNN_front(front)
    return front

def generate_KNN_front(front):
    # implements svms for each point on the pareto front
    # returns a list of SVMs
    models = []
    for individual in front:
        clf = KNeighborsClassifier(n_neighbors = individual[1][0])
        models += [clf]
    return models
