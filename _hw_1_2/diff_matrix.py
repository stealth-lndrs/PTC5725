
import numpy as np

def barycentric_weights(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.ones(n, dtype=float)
    for j in range(n):
        diff = x[j] - x
        diff[j] = 1.0
        w[j] = 1.0 / np.prod(diff)
    return w

def diff_matrix(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    w = barycentric_weights(x)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :]) + D[i, i]
    return D

def differentiate(x, fvals, k=1):
    x = np.asarray(x, dtype=float)
    fvals = np.asarray(fvals, dtype=float)
    D = diff_matrix(x)
    M = np.eye(len(x))
    for _ in range(k):
        M = M @ D
    return M @ fvals
