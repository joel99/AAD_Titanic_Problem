import parser
import svm
import rfClassifier
import knnClassifier
from sklearn.naive_bayes import GaussianNB
import knnClassifier
import matplotlib.pyplot as plt
import numpy as np

# Evaluate classifier
def get_fp_fn(clf, train, test):
    x_train, y_train = train
    x_test, y_test = test
    clf.fit(x_train, y_train)

    # predict the response
    preds = clf.predict(x_test)
    fp = sum([1 for i in range(len(preds)) if preds[i] == 1 and y_test[i][1] == 0])
    fn = sum([1 for i in range(len(preds)) if preds[i] == 0 and y_test[i][1] == 1])
    return (fp, fn)

def main():
    train, test = parser.load_split_all()
    train_x, train_y = train
    test_x, test_y = test

    rf_front = rfClassifier.find_best_rf(train_x, train_y)
    svm_front = svm.find_best_SVM(train_x, train_y)
    knn_front = knnClassifier.find_best_knn(train_x, train_y)
    gnb_front = [GaussianNB()]

    rf_scores = np.asarray([get_fp_fn(clf, train, test) for clf in rf_front])
    svm_scores = np.asarray([get_fp_fn(clf, train, test) for clf in svm_front])
    knn_scores = np.asarray([get_fp_fn(clf, train, test) for clf in knn_front])
    gnb_scores = np.asarray([get_fp_fn(clf, train, test) for clf in gnb_front])
    # Score shape is Nx2 - I need true front due to random results
    rf_true = []
    for score in rf_scores:
        score = [score, ()] # dummy
        rf_true = parser.update_front(rf_true, score, parser.pareto_dominance_min)
    rf_scores = np.asarray([np.asarray(ind[0]) for ind in rf_true])
    svm_true = []
    for score in svm_scores:
        score = [score, ()]
        svm_true = parser.update_front(svm_true, score, parser.pareto_dominance_min)
    svm_scores = np.asarray([np.asarray(ind[0]) for ind in svm_true])
    knn_true = []
    for score in knn_scores:
        score = [score, ()]
        knn_true = parser.update_front(knn_true, score, parser.pareto_dominance_min)
    knn_scores = np.asarray([np.asarray(ind[0]) for ind in knn_true])

    print("Summary of fronts")
    print(rf_scores)
    print(svm_scores)
    print(knn_scores)
    print(gnb_scores)
    # Sort scores so they display pseudo HoF
    rf_scores = rf_scores[np.argsort(rf_scores[:, 0])]
    svm_scores = svm_scores[np.argsort(svm_scores[:, 0])]
    knn_scores = knn_scores[np.argsort(knn_scores[:, 0])]

    fig, ax = plt.subplots()
    ax.set_title("Titanic Pareto Fronts")
    ax.plot(rf_scores[:,0], rf_scores[:,1], c='b', marker='o', markersize='12', label='RF')
    ax.plot(svm_scores[:, 0], svm_scores[:, 1], c='g', marker='o', markersize='12', label='SVM')
    ax.plot(knn_scores[:, 0], knn_scores[:, 1], c='r', marker='o', markersize='12', label='KNN')
    ax.plot(gnb_scores[:,0], gnb_scores[:,1], c='m', marker='o', markersize='12', label='GNB')
    # ax.plot(gnb_scores[:, 0], gnb_scores[:, 1], c='m', label='GNB')
    plt.xlabel("False Positives")
    plt.ylabel("False Negatives")
    ax.legend()
    plt.show()

main()
