from Tool import load_data, get_acc
import numpy as np
from copy import deepcopy
from tqdm import tqdm

EPS = 1e-10

class Tree:
    def __init__(self, n_features):
        self.age = 0
        self.n_features = n_features
        self.weight = np.random.randint(low=0, high=2, size=(n_features))
        if len(self.index()) == 0: self.weight[0] = 1
        self.acc = 0

    def set(self, v):
        for i in range(self.n_features):
            self.weight[i] = v

    def update(self, X, y, esti="5nn"):
        self.acc = get_acc(X[:, self.index()], y, esti, True)

    def index(self):
        idx = []
        for i in range(self.n_features):
            if self.weight[i] > 0.9:
                idx.append(i)
        return idx

    def DR(self):
        return 1.0 - 1.0 * len(self.index()) / self.n_features

    def reverse(self, num):
        if num > self.n_features: num = self.n_features
        idx = np.array([x for x in range(self.n_features)])
        np.random.shuffle(idx)

        for _ in range(num):
            i = idx[_]
            self.weight[i] = 1.0 - self.weight[i]

        if len(self.index()) == 0: self.weight[0] = 1



def FSFOA(filepath, esti):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)

    # Settings
    life_time = 15
    area_limit = 50
    transfer_rate = 5. / 100
    lsc, gsc = int(n_features / 5), int(n_features * 2 / 5)

    # Init Forest
    forest = [Tree(n_features) for _ in range(int(area_limit/10))]
    acc_pool = []

    # Run Forest
    pbar = tqdm(range(100))
    for epoch in pbar:
        # Local Seeding
        new_trees = []
        for tree in forest:
            if tree.age > 0: continue
            new_trees.append(deepcopy(tree))
            new_trees[-1].reverse(lsc)
            new_trees[-1].update(X, y, esti)
            tree.age = tree.age + 1
        forest += new_trees

        # Delete Trees
        candidate, new_forest = [], []
        for tree in forest:
            if tree.age > life_time:
                candidate.append(tree)
            else:
                new_forest.append(tree)

        forest = []
        new_forest.sort(key=lambda x: x.acc, reverse=True)
        for i in range(len(new_forest)):
            if i >= area_limit:
                candidate.append(new_forest[i])
            else:
                forest.append(new_forest[i])

        # Global Seeding
        candidate_len = len(candidate)
        idx = np.array([x for x in range(candidate_len)])
        np.random.shuffle(idx)
        for ii in range(int(candidate_len*transfer_rate)):
            i = idx[ii]
            tree = candidate[i]
            tree.age = 0
            tree.reverse(gsc)
            tree.update(X, y, esti)
            forest.append(tree)

        forest.sort(key=lambda x: x.acc, reverse=True)
        forest[0].age = 0

        # print(forest[0].acc)

        if len(acc_pool) > 10: acc_pool.pop(0)
        if len(acc_pool) >= 10 and np.mean(acc_pool)+EPS >= forest[0].acc: break
        acc_pool.append(forest[0].acc)
        # print(len(new_forest), candidate_len, np.mean(acc_pool), acc_pool[-1])

    print("Acc = {0} DR = {1}".format(forest[0].acc, forest[0].DR()))
    return forest[0].weight



# print(FSFOA("dataset/vehicle.csv"))


