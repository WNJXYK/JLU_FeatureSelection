from sklearn.linear_model.logistic import LogisticRegression
from Tool import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def Lasso(filepath, esti):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)

    # Train
    weight = np.ones(n_features)
    pbar = tqdm(range(100))
    last = 0
    for epoch in pbar:
        logitModel = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=10000, penalty="l1", n_jobs=-1)
        logitModel.fit(X * weight, y.ravel())

        # Calc Weight
        cnt, weight = 0, []
        for i in range(n_features):
            if np.abs(logitModel.coef_[0][i])>0:
                weight.append(1)
            else:
                weight.append(0)
                cnt = cnt + 1
        weight = np.array(weight)

        if last != cnt:
            last = cnt
        else:
            break

    # print("Dr = {0}".format(1.0 * cnt / n_features))
    return weight


# Lasso("dataset/vehicle.csv")
