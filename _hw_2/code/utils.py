
import numpy as np

def cheb_lobatto_nodes(n, neg_flip=False):
    k = np.arange(0, n+1)
    x = np.cos(np.pi * k / n)
    if neg_flip:
        x = -x
    return x

def barycentric_weights(x):
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0/np.prod(diff)
    return w

def barycentric_eval(x_nodes, y_nodes, w, x_eval):
    x_nodes = np.asarray(x_nodes); y_nodes = np.asarray(y_nodes); w = np.asarray(w)
    x_eval = np.atleast_1d(x_eval)
    P = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        diff = x - x_nodes
        mask = np.isclose(diff, 0.0)
        if np.any(mask):
            P[i] = y_nodes[mask][0]
            continue
        terms = w / diff
        P[i] = np.dot(terms, y_nodes) / np.sum(terms)
    return P

def cheb_basis_matrix(n, neg_flip=False):
    x = cheb_lobatto_nodes(n, neg_flip=neg_flip)
    theta = np.arccos(x)
    B = np.zeros((n+1, n+1))
    for j in range(n+1):
        B[:, j] = np.cos(j*theta)
    return x, B

def cheb_diff_matrices(n):
    if n == 0:
        return np.array([[0.0]]), np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi*np.arange(n+1)/n)
    c = np.ones(n+1)
    c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**np.arange(n+1))
    X = np.tile(x,(n+1,1))
    dX = X - X.T + np.eye(n+1)
    D = (np.outer(c,1/c)) / dX
    D = D - np.diag(np.sum(D,axis=1))
    D2 = D @ D
    return D, D2, x

def dct_type1_via_fft(y):
    n = len(y) - 1
    if n == 0:
        return y.copy()
    z = np.concatenate([y, y[-2:0:-1]])
    Z = np.fft.fft(z)
    c = np.real(Z[:n+1])
    c[0] = c[0]/2.0
    c[-1] = c[-1]/2.0
    return c

def cheb_reconstruct_from_coeffs(x, c):
    theta = np.arccos(x)
    vals = np.zeros_like(x, dtype=float)
    for k, ck in enumerate(c):
        vals += ck*np.cos(k*theta)
    return vals

def clenshaw_curtis_weights(n):
    if n == 1:
        return np.array([1.0, 1.0])
    c = np.zeros(n+1)
    c[0] = 2; c[-1] = 2
    for k in range(1, n//2 + 1):
        v = 2.0/(1 - (2*k)**2)
        c[0] += v
        c[-1] += v
        for j in range(1, n):
            c[j] += v*np.cos(2*k*np.pi*j/n)
    w = c / n
    return w
