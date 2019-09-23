from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time


def load_data(filepath):
    # Read File
    raw = np.array(pd.read_csv(filepath, header = None))
    n_samples, n_features = raw.shape[0], raw.shape[1] - 1
    X, y = np.array(raw[:,:n_features]), np.array(raw[:, n_features]).reshape((-1, 1))
    X = scale(X)
    n_classes = len(set(raw[:, n_features]))

    return X, y, (n_samples, n_features, n_classes)


def get_acc(X, y, esti='5nn', nofold=True):
    # Build Model
    model = KNeighborsClassifier(n_jobs=-1)
    if esti[1:] == 'nn': model = KNeighborsClassifier(n_neighbors=int(esti[0]), n_jobs=-1)
    if esti == 'svm': model = SVC(kernel="rbf", gamma='auto')
    if esti == 'cart': model = DecisionTreeClassifier()

    # Calc
    acc = []
    if nofold:
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=19260817)
        model.fit(train_X, train_y.ravel())
        acc.append(accuracy_score(model.predict(test_X), test_y))
    else:
        kf = KFold(n_splits=10, random_state=19260817)
        for train_index, test_index in kf.split(X):
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            model.fit(train_X, train_y.ravel())
            acc.append(accuracy_score(model.predict(test_X), test_y))

    return np.mean(acc)


def Test(func, fp=None):
    files = ["cleveland", "ionosphere", "heart", "vehicle", "LSVT", "srbct", "arcene"]
    if fp is not None: fp = open(fp, "w+")

    for name in files:
        path = "dataset/" + name + ".csv"
        X, y, (n_samples, n_features, _) = load_data(path)

        print("Dataset {0} ({1})".format(name, n_samples))
        if fp is not None:
            print("Dataset {0} ({1})".format(name, n_samples), file=fp)

        start_time = time.time()
        weight = func(path)
        idx = []
        for i in range(n_features):
            if weight[i] > 0.9: idx.append(i)

        A0 = get_acc(X[:, idx], y, "1nn", False)
        A1 = get_acc(X[:, idx], y, "5nn", False)
        A2 = get_acc(X[:, idx], y, "svm", False)
        A3 = get_acc(X[:, idx], y, "cart", False)
        B0 = get_acc(X[:, idx], y, "1nn", True)
        B1 = get_acc(X[:, idx], y, "5nn", True)
        B2 = get_acc(X[:, idx], y, "svm", True)
        B3 = get_acc(X[:, idx], y, "cart", True)

        print("Time = {1} DR = {0}".format(1. - 1. * len(idx) / len(weight), time.time()-start_time))

        print(" * 1nn & 10-fold {0}".format(A0))
        print(" * 5nn & 10-fold {0}".format(A1))
        print(" * SVM & 10-fold {0}".format(A2))
        print(" * Cart & 10-fold {0}".format(A3))

        print(" * 1nn & 70%-30% {0}".format(B0))
        print(" * 5nn & 70%-30% {0}".format(B1))
        print(" * SVM & 70%-30% {0}".format(B2))
        print(" * Cart & 70%-30% {0}".format(B3))

        if fp is not None:
            print("Time = {1} DR = {0}".format(1. - 1. * len(idx) / len(weight), time.time()-start_time), file=fp)
            print(" * 1nn & 10-fold {0}".format(A0), file=fp)
            print(" * 5nn & 10-fold {0}".format(A1), file=fp)
            print(" * SVM & 10-fold {0}".format(A2), file=fp)
            print(" * Cart & 10-fold {0}".format(A3), file=fp)

            print(" * 1nn & 70%-30% {0}".format(B0), file=fp)
            print(" * 5nn & 70%-30% {0}".format(B1), file=fp)
            print(" * SVM & 70%-30% {0}".format(B2), file=fp)
            print(" * Cart & 70%-30% {0}".format(B3), file=fp)
            fp.flush()

    if fp is not None: fp.close()
