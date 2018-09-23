import parser
import svm
import rfClassifier

def main():
    train, test = parser.load_split_all()
    train_x, train_y = train
    test_x, test_y = test
    best_rf = rfClassifier.find_best_rf(train_x, train_y)
    svm_front = svm.find_best_SVM(train_x, train_y)
    best_knn = knnClassifier.find_best_knn(train_x, train_y)

main()
