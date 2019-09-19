from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def get_acc(X, y, esti='5nn', flag = True):
    kf = KFold(n_splits=2 if flag else 10)
    acc = []

    for train_index, test_index in kf.split(X):
        if flag:
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=19260817)
        else:
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]

        model = KNeighborsClassifier()
        if esti[1:] == 'nn': model = KNeighborsClassifier(n_neighbors=int(esti[0]))
        if esti == 'svm': model = SVC(kernel="rbf", gamma='auto')
        model.fit(train_X, train_y.ravel())
        acc.append(accuracy_score(model.predict(test_X), test_y))

    return np.mean(acc)