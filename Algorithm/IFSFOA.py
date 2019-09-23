from Algorithm.FSFOA import Tree
from Tool import load_data
import numpy as np
from copy import deepcopy
from tqdm import tqdm



def IFSFOA(filepath):
    # Input
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)

    # Settings
    life_time = 15
    area_limit = 50
    transfer_rate = 5. / 100
    lsc, gsc = int(n_features / 5), int(n_features * 2 / 5)

    # Init Forest
    forest = []
    for i in range(area_limit):
        tree = Tree(n_features)
        if i < area_limit/5:
            tree.set(1)
            tree.reverse(lsc)
        else:
            tree.set(0)
            tree.reverse(lsc)
        forest.append(tree)

    # Run Forest
    pbar = tqdm(range(100))
    acc_pool= []
    for epoch in pbar:
        # Local Seeding
        new_trees = []
        for tree in forest:
            tt = deepcopy(tree)
            for i in range(lsc):
                tn = deepcopy(tt)
                tn.reverse(lsc)
                tn.update(X, y)
                if tn.acc > tt.acc:
                    new_trees.append(tn)
                    tt = tn
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
        new_forest.sort(key=lambda x: (x.acc, -len(x.index())), reverse=True)
        for i in range(len(new_forest)):
            if i >= area_limit:
                candidate.append(new_forest[i])
            else:
                forest.append(new_forest[i])

        # Global Seeding
        candidate_len = len(candidate)
        idx = np.array([x for x in range(candidate_len)])
        np.random.shuffle(idx)
        for i in range(int(candidate_len*transfer_rate)):
            tree = candidate[i]
            tree.age = 0
            tree.reverse(gsc)
            tree.update(X, y)
            forest.append(tree)

        forest.sort(key=lambda x: (x.acc, -len(x.index())), reverse=True)
        forest[0].age = 0

        # print(forest[0].acc)
        acc_pool.append(forest[0].acc)
        if len(acc_pool) > 10: acc_pool.pop(0)
        if len(acc_pool) >= 10 and np.mean(acc_pool) >= forest[0].acc: break

    print("Acc = {0} DR = {1}".format(forest[0].acc, forest[0].DR()))
    return forest[0].weight

# print(IFSFOA("dataset/vehicle.csv"))