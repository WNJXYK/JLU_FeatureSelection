from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
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
    kf = KFold(n_splits=2 if nofold else 10)
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
        model.fit(train_X, train_y.ravel())
        acc.append(accuracy_score(model.predict(test_X), test_y))

    return np.mean(acc)

def test(func):
    files = ["arcene", "LVST", "cleveland", "ionosphere", "vehicle"]

    for name in files:
        path = "dataset/" + files + ".csv"
        func