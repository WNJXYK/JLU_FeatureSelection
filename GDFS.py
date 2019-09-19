import tensorflow as tf
from sklearn.linear_model.logistic import LogisticRegression
from CheckAcc import get_acc
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.model_selection import train_test_split
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

    return X, y, None, (n_samples, n_features, n_classes)


def build_tf_graph(n_features, n_classes, weight_init):
    n_weight = 1 if n_classes == 2 else n_classes
    X_input = tf.placeholder(tf.float32, shape=(None, n_features))
    y_input = tf.placeholder(tf.float32, shape=(None, n_classes))
    coef = tf.placeholder(tf.float32, shape=(n_features, n_weight))
    bias = tf.placeholder(tf.float32, shape=(n_weight))
    lam = tf.placeholder(tf.float32)
    learning = tf.placeholder(tf.float32)

    weight = tf.Variable(weight_init, dtype=tf.float32, name="feature_weight")
    weight = weight * n_features / tf.reduce_sum(weight)

    X_trans = tf.multiply(X_input, weight)
    print(X_trans, coef)
    y_proba = 1. / (1. + tf.exp(-(tf.matmul(X_trans, coef) + bias)))
    y_proba /= tf.reduce_sum(y_proba)

    mse_loss = tf.reduce_mean(tf.square(y_proba - y_input))
    l1_loss = lam * tf.reduce_sum(tf.exp(-100. * (weight - 0.1)))
    loss = mse_loss + l1_loss
    train = tf.train.AdamOptimizer(learning).minimize(loss)

    return (X_input, y_input, coef, bias, lam, learning), (train, weight, loss)




def GDFS(filepath):
    # Input
    X, y, y_onehot, (n_samples, n_features, n_classes) = load_data(filepath)

    EPS = 1.0
    weight_val = np.ones(n_features)
    (X_input, y_input, coef, bias, lam, learning), (train, weight, loss) = build_tf_graph(n_features, n_classes, weight_val)

    with tf.Session() as sess:
        for rept in range(10):
            sess.run(tf.global_variables_initializer())
            last_acc, fixed_weight_val = 0, np.ones(n_features)
            best_acc, best_dr, best_weight_val = 0, 0, np.ones(n_features)
            for R in range(100):
                X_trans = X * fixed_weight_val
                X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=19260817)
                logitModel = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=10000, penalty="l2")
                logitModel.fit(X_train, y_train.ravel())

                y_prob = logitModel.predict_proba(X)
                _, loss_val, weight_val = sess.run([train, loss, weight],
                                                   feed_dict={X_input: X,
                                                               y_input: y_prob,
                                                               coef: logitModel.coef_.T,
                                                               bias: logitModel.intercept_,
                                                               learning: 0.1,
                                                               lam: 1})
                # print("Loss = {0}".format(loss_val))
                ori_wei = np.array(weight_val)
                idx = []
                for i in range(len(weight_val)):
                    if weight_val[i] < EPS:
                        weight_val[i] = 0
                    else:
                        weight_val[i] = 1
                        idx.append(i)

                cur_acc = get_acc(X[:, idx], y, "1nn", True)
                cur_dr = 1.0 - 1.0 * len(idx) / n_features

                if (cur_acc > best_acc or (np.abs(cur_acc-best_acc) < 1e-8 and cur_dr > best_dr)) and cur_dr > 0:
                    best_acc, best_dr = cur_acc, cur_dr
                    best_weight_val = weight_val

                if cur_acc > last_acc or np.random.rand() < 1/(R+1):
                    print("  * Acc = {0} DR = {1}".format(cur_acc, cur_dr))
                    fixed_weight_val = weight_val
                    last_acc = cur_acc

            print("Best Acc = {0}, DR = {1}".format(best_acc, best_dr))

GDFS("dataset/arcene.csv")


