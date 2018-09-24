import parser
import svm
import rfClassifier
import knnClassifier

def main():
    train, test = parser.load_split_all()
    train_x, train_y = train
    test_x, test_y = test
    # rf_front = rfClassifier.find_best_rf(train_x, train_y)
    svm_front = svm.find_best_SVM(train_x, train_y)
    knn_front = knnClassifier.find_best_knn(train_x, train_y)
    # classifier_list = [svm_front, knn_front, rf_front, gnb_front]
main()