import tensorflow as tf
from sklearn.linear_model.logistic import LogisticRegression
from Tool import get_acc, load_data
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

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
    y_proba = 1. / (1. + tf.exp(-(tf.matmul(X_trans, coef) + bias)))
    y_proba /= tf.reduce_sum(y_proba)

    mse_loss = tf.reduce_mean(tf.square(y_proba - y_input))
    l1_loss = tf.reduce_sum(tf.exp(-100. * (weight - 0.1)))
    loss = mse_loss + lam * l1_loss
    train = tf.train.AdamOptimizer(learning).minimize(loss)

    return (X_input, y_input, coef, bias, lam, learning), (train, weight, loss)

def GDFS(filepath):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)

    EPS = 1.0
    weight_val = np.ones(n_features)
    (X_input, y_input, coef, bias, lam, learning), (train, weight, loss) = build_tf_graph(n_features, n_classes, weight_val)

    best_acc, best_dr, best_weight_val = 0, 0, np.ones(n_features)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pbar = tqdm(range(100))
        for epoch in pbar:
            # Logistic Regression
            X_trans = X * weight_val
            X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=19260817)
            logitModel = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=10000, penalty="l2")
            logitModel.fit(X_train, y_train.ravel())

            # Lasso Features with Proba
            y_prob = logitModel.predict_proba(X)
            _, loss_val, weight_val = sess.run([train, loss, weight],
                                               feed_dict={X_input: X,
                                                           y_input: y_prob,
                                                           coef: logitModel.coef_.T,
                                                           bias: logitModel.intercept_,
                                                           learning: 0.1,
                                                           lam: 1})

            # Delete Features
            idx = []
            for i in range(len(weight_val)):
                if weight_val[i] < EPS:
                    weight_val[i] = 0
                else:
                    weight_val[i] = 1
                    idx.append(i)
            if len(idx) == 0: break

            # Calc Acc, Dr
            cur_acc = get_acc(X[:, idx], y, "cart", False)
            cur_dr = 1.0 - 1.0 * len(idx) / n_features
            # print("  * Acc = {0} DR = {1}".format(cur_acc, cur_dr))

            # Update Best Weight (Acc First)
            if (cur_acc > best_acc or (np.abs(cur_acc-best_acc) < 1e-8 and cur_dr > best_dr)) and cur_dr > 0:
                best_acc, best_dr = cur_acc, cur_dr
                best_weight_val = weight_val
                # print(" * Update Acc={0} Dr={1}".format(best_acc, best_dr))

    print("Best Acc = {0}, DR = {1}".format(best_acc, best_dr))
    return best_weight_val

GDFS("dataset/vehicle.csv")
