from Tool import load_data, get_acc, Test
import numpy as np
import random
import operator
import copy
import math

class Tree:
    def __init__(self, n_features):
        self.age = 0
        self.n_features = n_features
        self.gsc = 2
        self.lsc = 2
        self.f = np.random.randint(2, size=n_features)
        self.fitness = 0

    def set(self, p, v):
        self.f[p] = v

    def change(self, p):
        self.f[p] = self.f[p] ^ 1


def cal_fitness(X, y, tree, a, b, esti):
    #esti = "svm"
    idx = []
    for i in range(tree.n_features):
        if tree.f[i] == 1:
            idx.append(i)
    A0 = get_acc(X[:, idx], y, esti, True)
    return A0*a+b


def get_igr(X, y, n_features):

    a = np.unique(y)
    n = len(a)
    length = len(y)
    cnt = []
    for i in range(n):
        cnt.append(0)

    for i in range(length):
        for j in range(n):
            if y[i] == a[j]:
                cnt[j] = cnt[j]+1

    Hc = 0
    for i in range(n):
        Hc = Hc - cnt[i] / length * math.log(float(cnt[i]) / float(length), 2)

    HX = 0
    IGR = []

    Data = X.copy()
    for i in range(n_features):
        a = Data[:, i]
        mi = a.mean()
        cnt = []
        HcX = 0
        for j in range(n):
            cnt.append(0)
        for j in range(len(Data)):
            if Data[j][i] >= mi:
                Data[j][i] = 1
                for k in range(n):
                    if y[j] == a[k]:
                        cnt[k] = cnt[k]+1
            else:
                Data[j][i] = 0
        for j in range(n):
            if sum(cnt) != 0 & cnt[j] != 0:
                HcX = HcX - cnt[j] / sum(cnt) * math.log(float(cnt[j]) / float(sum(cnt)), 2)
        IGR.append(Hc-HcX)
        s = Data[:, i].sum()
        HX = HX - s/len(Data)*math.log(s/len(Data), 2)

    for i in range(n_features):
        IGR[i] = IGR[i]/HX
    return IGR.index(max(IGR))


def tree_cmp(treea, forest, candidate):
    i = 0
    for tree in forest:
        mark = 0
        for i in range(tree.n_features):
            if treea.f[i] != tree.f[i]:
                mark = 1
                break
        if mark == 0:
            return i
        i = i+1

    for tree in candidate:
        mark = 0
        for i in range(tree.n_features):
            if treea.f[i] != tree.f[i]:
                mark = 1
                break
        if mark == 0:
            return i
        i = i + 1
    return -1


def EFSFOA(filepath, esti):
#init
    X, y, (n_samples, n_features, n_classes) = load_data(filepath)
    transferrate = 0.05
    arealimit = 50
    lifetime = 15
    lsc = 2
    best = 0
    forest = []
    candidate = []
    n = 5
    igr = np.random.randint(n_features)
    igr = get_igr(X,y,n_features)
    cmpfun = operator.attrgetter('fitness')
    cnt = 0
    alpha = 100.0
    beta = 0.01

#init forest

    for i in range(n):
        tree = Tree(n_features)
        tree.set(igr,1)
        forest.append(tree)

    changefe = random.sample(range(0, n), int(n / 2))
    for i in range(int(n / 2)):
        forest[changefe[i]].set(igr, 1)

    for i in range(n):
        forest[i].fitness = cal_fitness(X,y,forest[i],alpha,beta*np.sum(forest[i].f)/n_features, esti)

#loop
    while cnt < 10:
        #seeding near
        n = len(forest)
        i = 0
        while i < n:
            if forest[i].age >= lifetime:
                candidate.append(forest[i])
                forest.pop(i)
                n -= 1
                i -= 1
            elif forest[i].age == 0:
                seed = random.sample(range(0, n_features), lsc)
                for j in range(lsc):
                    tree = copy.deepcopy(forest[i])
                    tree.change(seed[j])
                    if np.sum(tree.f) != 0:
                        k = tree_cmp(tree,forest,candidate)
                        if k == -1:
                            tree.fitness = cal_fitness(X,y,tree,alpha,beta*np.sum(tree.f)/n_features, esti)
                            forest.append(tree)
                        elif k < len(forest):
                            forest[k].age = forest[k].age+1
            forest[i].age = forest[i].age + 1
            i += 1
        n = len(forest)
        if n > arealimit:
            forest.sort(key=cmpfun)
            for i in range(n - arealimit):
                candidate.append(forest[0])
                forest.pop(0)
        if best >= forest[len(forest) - 1].fitness:
            cnt = cnt+1
        else:
            cnt = 0

        best = max(best, forest[len(forest) - 1].fitness)
        #print(best, cnt)
        #print("########################")
        #seeding global
        n = len(candidate)
        num = int(transferrate*n)
        c_candidate = random.sample(range(0, n), num)
        for i in range(num):
            tree = candidate[c_candidate[i]]
            tree.gsc = int(min(2+2*lifetime, n_features*0.5)/(tree.age+1))
            c_feature = random.sample(range(0, n_features), tree.gsc)
            for j in range(tree.gsc):
                tree.change(c_feature[j])
            tree.fitness = cal_fitness(X, y, tree, alpha, beta * np.sum(tree.f) / n_features, esti)

        if num > 0:
            candidate.sort(key=cmpfun, reverse=True)
            candidate[0].age = 0
            forest.append(candidate[0])
            candidate.pop(0)

    forest.sort(key=cmpfun)
    su = []
    second = []
    third = []
    for i in range(n_features):
        su.append([0, i])
        second.append(0)
        third.append(0)

    for tree in forest:
        for i in range(n_features):
            su[i][0] = su[i][0]+tree.f[i]

    su = sorted(su)
    su.reverse()
    n = forest[len(forest)-1].f.sum()

    tree2 = Tree(n_features)
    tree3 = Tree(n_features)

    for i in range(n):
        tree2.f[su[i][1]] = 1
        if i != n-1:
            tree3.f[su[i][1]] = 1

    c1 = cal_fitness(X,y,forest[len(forest)-1],alpha,beta*np.sum(forest[len(forest)-1].f)/n_features, esti)
    c2 = cal_fitness(X,y,tree2,alpha,beta*np.sum(tree2.f)/n_features, esti)
    c3 = cal_fitness(X,y,tree3,alpha,beta*np.sum(tree3.f)/n_features, esti)

    if c2 < c1:
        tree2 = copy.deepcopy(forest[len(forest)-1])
        c2 = c1
    if c3 < c2:
        tree3 = copy.deepcopy(tree2)
        c3 = c2
    #print("DR: {0}".format(1.-np.sum(tree3.f) / n_features))
    return tree3.f


#EFSFOA("dataset/vehicle.csv")
Test(EFSFOA, "EFSFOA.txt")
