from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def load_data(filepath):
    # Read File
    raw = np.array(pd.read_csv(filepath, header = None))
    n_samples, n_features = raw.shape[0], raw.shape[1] - 1
    X, y = np.array(raw[:,:n_features]), np.array(raw[:, n_features]).reshape((-1, 1))
    X = scale(X)
    n_classes = len(set(raw[:, n_features]))

    return X, y, (n_samples, n_features, n_classes)

def get_acc(X, y, esti='5nn', nofold=True):
    kf = KFold(n_splits=2 if nofold else 10, random_state=19260817)
    acc = []

    for train_index, test_index in kf.split(X):
        if nofold:
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=19260817)
        else:
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]

        model = KNeighborsClassifier()
        if esti[1:] == 'nn': model = KNeighborsClassifier(n_neighbors=int(esti[0]))
        if esti == 'svm': model = SVC(kernel="rbf", gamma='auto')
        if esti == 'cart': model = DecisionTreeClassifier()
        model.fit(train_X, train_y.ravel())
        acc.append(accuracy_score(model.predict(test_X), test_y))

    return np.mean(acc)

def test(func):
    files = ["cleveland", "ionosphere", "heart", "vehicle", "LSVT", "srbct", "arcene"]

    for name in files:
        print("Dataset {0}".format(name))

        path = "dataset/" + name + ".csv"
        X, y, (_, n_features, _) = load_data(path)

        weight = func(path)
        idx = []
        for i in range(n_features):
            if weight[i] > 0.9: idx.append(i)

        A1 = get_acc(X[:, idx], y, "5nn", False)
        A2 = get_acc(X[:, idx], y, "svm", False)
        A3 = get_acc(X[:, idx], y, "cart", False)
        B1 = get_acc(X[:, idx], y, "5nn", True)
        B2 = get_acc(X[:, idx], y, "svm", True)
        B3 = get_acc(X[:, idx], y, "cart", True)


        print(" * 5nn & 10-fold {0}".format(A1))
        print(" * SVM & 10-fold {0}".format(A2))
        print(" * Cart & 10-fold {0}".format(A3))
        print(" * 5nn & 70%-30% {0}".format(B1))
        print(" * SVM & 70%-30% {0}".format(B2))
        print(" * Cart & 70%-30% {0}".format(B3))
