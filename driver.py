import parser
import svm
import rfClassifier
import knnClassifier
from sklearn.neighbors import KNeighborsClassifier

test_label_fn = 'gender_submission.csv'

def main():
    train, test = parser.load_split_all()
    train_x, train_y = train
    test_x, test_y = test

    rf_front = rfClassifier.find_best_rf(train_x, train_y)
    svm_front = svm.find_best_SVM(train_x, train_y)
    knn_front = knnClassifier.find_best_knn(train_x, train_y)
    # classifier_list = [svm_front, knn_front, rf_front, gnb_front]
# main()

def get_fp_fn(clf):
    ((x_train, y_train), (x_test, y_test)) = parser.load_split_all()
    clf.fit(x_train, y_train)

    # predict the response
    preds = clf.predict(x_test)
    fp = sum([1 for i in range(len(preds)) if preds[i] == 1 and y_test[i] == 0])
    fn = sum([1 for i in range(len(preds)) if preds[i] == 0 and y_test[i] == 1])
    return (fp, fn)
get_fp_fn(KNeighborsClassifier(n_neighbors=5))