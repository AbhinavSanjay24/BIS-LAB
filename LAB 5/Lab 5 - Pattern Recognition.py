import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

def create_nests(n_nests, n_clusters, X):
    nests = []
    for _ in range(n_nests):
        idx = np.random.choice(len(X), n_clusters, replace=False)
        nests.append(X[idx].copy())
    return np.array(nests)

def fitness(nest, X):
    labels, min_dist = pairwise_distances_argmin_min(X, nest)
    return np.sum(min_dist)

def abandon_nests(nests, abandon_prob, X):
    for i in range(len(nests)):
        if np.random.rand() < abandon_prob:
            idx = np.random.choice(len(X), nests.shape[1], replace=False)
            nests[i] = X[idx].copy()
    return nests

def cuckoo_search(X, n_clusters=2, n_nests=5, n_iter=10, abandon_prob=0.3):
    nests = create_nests(n_nests, n_clusters, X)
    best_fitness = float('inf')
    best_nest = None
    for it in range(n_iter):
        for i in range(n_nests):
            nest_new = nests[i] + 0.1 * np.random.randn(*nests[i].shape)
            fit_new = fitness(nest_new, X)
            fit_current = fitness(nests[i], X)
            if fit_new < fit_current:
                nests[i] = nest_new
            if fit_new < best_fitness:
                best_fitness = fit_new
                best_nest = nest_new
        nests = abandon_nests(nests, abandon_prob, X)
    return best_nest, best_fitness

X, _ = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
best_nest, best_fitness = cuckoo_search(X)
print(best_nest, best_fitness)
