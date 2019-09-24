from sklearn.linear_model.logistic import LogisticRegression
from Tool import load_data, get_acc, Test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Process, Pool
import time

EPS = 1e-12

class Node:
    def __init__(self, n_features, one_flag = False):
        self.n_features = n_features
        if one_flag:
            self.weight = np.ones(n_features)
        else:
            self.weight = np.zeros(n_features)
        self.index = []
        self.acc = self.dr = 0

    def update_index(self):
        self.index = []
        for i in range(self.n_features):
            if self.weight[i] > 0.9: self.index.append(i)

    def update(self, X, y, esti):
        self.dr = 1.0 - 1. * len(self.index) / self.n_features
        if len(self.index) == 0:
            self.acc = 0
        else:
            self.acc = get_acc(X[:, self.index], y, esti, True)

    def cross(self, partner):
        assert self.n_features == partner.n_features
        ret = Node(self.n_features)
        for i in range(self.n_features):
            if self.weight[i] > 0.9 or partner.weight[i] > 0.9:
                ret.weight[i] = 1
        ret.update_index()
        return ret

    def mutation(self, T):
        lim = int(min(T, self.n_features))
        for i in range(lim):
            idx = int(np.random.randint(0, self.n_features))
            if np.abs(self.weight[idx]) < EPS:
                self.weight[idx] = 1
        self.update_index()

    def fitness(self):
        return self.acc + 0.1 * self.dr

def lasso_task(x, X, y, esti):
    logitModel = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=10000, penalty="l1", n_jobs=1)
    logitModel.fit(X[:, x.index], y.ravel())
    for i in range(len(x.index)):
        if np.abs(logitModel.coef_[0][i]) < EPS:
            x.weight[x.index[i]] = 0
    x.update_index()
    x.update(X, y, esti)
    return x

def mutation_task(x, T):
    x.mutation(T)
    return x

def cross_task(x, father):
    return x.cross(father)

def LGA(filepath, esti, epoch_limit=100):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)
    task_pool = Pool(14)

    # Init Group
    group = []
    for k in range(1048576):
        if pow(2, k) > n_features: break
        white, black = Node(n_features), Node(n_features)
        for i in range(n_features):
            if ((i>>k)&1) == 1:
                black.weight[i] = 1
            else:
                white.weight[i] = 1
        black.update_index()
        group.append(black)
        white.update_index()
        group.append(white)
    group = task_pool.map(partial(lasso_task, X=X, y=y, esti=esti), group)
    group.sort(key=lambda x: x.fitness(), reverse=True)
    print("Inited")

    group_siz = int(len(group) * 1.5)
    fitness_pool = []
    epochs = tqdm(range(epoch_limit))
    for epoch in epochs:
        cur_group_siz = len(group)

        # Mutation
        mutation_T = max(1., n_features * np.log2(2. * (epoch_limit - epoch) / epoch_limit))
        mutation_group = task_pool.map(partial(mutation_task, T=mutation_T), group)
        mutation_group = task_pool.map(partial(lasso_task, X=X, y=y, esti=esti), mutation_group)
        print("Mutated")

        # Cross
        kf = KFold(5, True)
        lim, cur_cross = int(min(5, cur_group_siz)), 0
        cross_list, cross_group = [x for x in range(cur_group_siz)], []
        for _, cross_index in kf.split(cross_list):
            cross_obj = [group[x] for x in cross_index]
            cross_group += task_pool.map(partial(cross_task, father=group[cur_cross]), cross_obj)
            cur_cross = cur_cross + 1
        cross_group = task_pool.map(partial(lasso_task, X=X, y=y, esti=esti), cross_group)
        print("Crossed")

        # Select
        group += mutation_group
        group += cross_group
        group.sort(key=lambda x: x.fitness(), reverse=True)
        group = group[:group_siz]

        # Early Stop
        if len(fitness_pool) > 20: fitness_pool.pop(0)
        if len(fitness_pool) >= 20 and np.mean(fitness_pool)+EPS >= group[0].fitness(): break
        fitness_pool.append(group[0].fitness())

    task_pool.close()
    # Log
    print(group[0].fitness(), group[0].acc)

    return group[0].weight

if __name__ == "__main__":
    LGA("../dataset/vehicle.csv", "svm")