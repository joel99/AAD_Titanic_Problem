import parser
import svm
import rfClassifier
import knnClassifier
import gnb
import knnClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


test_label_fn = 'gender_submission.csv'

def get_fp_fn(clf):
    ((x_train, y_train), (x_test, y_test)) = parser.load_split_all()
    clf.fit(x_train, y_train)

    # predict the response
    preds = clf.predict(x_test)
    fp = sum([1 for i in range(len(preds)) if preds[i] == 1 and y_test[i][0] == 0])
    fn = sum([1 for i in range(len(preds)) if preds[i] == 0 and y_test[i][0] == 1])
    return (fp, fn)

def main():
    train, test = parser.load_split_all()
    train_x, train_y = train
    test_x, test_y = test
    (fig, sc) = parser.init_graph()

    rf_front = rfClassifier.find_best_rf(train_x, train_y)
    svm_front = svm.find_best_SVM(train_x, train_y)
    knn_front = knnClassifier.find_best_knn(train_x, train_y)
    gnb_front = [gnb.gaussianNB()]
    for clf in rf_front:
        rf_fp, rf_fn = get_fp_fn(clf)
    for clf in svm_front:
        svm_fp, svm_fn = get_fp_fn(clf)
    for clf in knn_front:
        knn_fp, knn_fn = get_fp_fn(clf)
    for clf in gnb_front:
        gnb_fp, gnb_fn = get_fp_fn(clf)
    fig, ax = plt.subplots()
    ax.set_title("Titanic Pareto Fronts")
    ax.plot(rf_fp, rf_fn, c='b', label='Random Forest')
    ax.plot(svm_fp, svm_fn, c='g', label='SVM')
    ax.plot(knn_fp, knn_fn, c='r', label='KNN')
    ax.plot(gnb_fp, gnb_fn, c='m', label='GNB')
    plt.xlabel("False Positives")
    plt.ylabel("False Negatives")
    ax.legend()
    plt.show()
main()

