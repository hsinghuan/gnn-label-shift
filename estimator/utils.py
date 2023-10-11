import numpy as np

def calculate_marginal(y, n_classes):
    mu = np.zeros(shape=(n_classes, 1))
    for i in range(n_classes):
        mu[i] = np.sum(y == i)
    return mu / y.shape[0]