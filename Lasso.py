from sklearn.linear_model.logistic import LogisticRegression
from Tool import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def Lasso(filepath):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)

    # Train
    pbar = tqdm(range(1))
    for epoch in pbar:
        logitModel = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=10000, penalty="l1")
        logitModel.fit(X, y.ravel())

        # Calc Weight
        cnt, weight = 0, []
        for i in range(n_features):
            if np.abs(logitModel.coef_[0][i])>0:
                weight.append(1)
            else:
                weight.append(0)
                cnt = cnt + 1
        weight = np.array(weight)

    print("Dr = {0}".format(1.0 * cnt / n_features))
    return weight


# Lasso("dataset/vehicle.csv")
