
import numpy as np

def bary_weights_generic(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.ones(n, dtype=float)
    for j in range(n):
        diff = x[j] - x
        diff[j] = 1.0
        w[j] = 1.0 / np.prod(diff)
    return w

def bary_weights_cheb2(n):
    w = np.ones(n+1, dtype=float)
    for j in range(n+1):
        w[j] = (-1.0)**j
    w[0] *= 0.5
    w[-1] *= 0.5
    return w

def bary_eval(x, y, xq, w=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)

    if w is None:
        w = bary_weights_generic(x)
    else:
        w = np.asarray(w, dtype=float)

    yq = np.empty_like(xq, dtype=float)
    for idx, z in enumerate(xq):
        diffs = z - x
        mask = np.isclose(diffs, 0.0, atol=0.0, rtol=0.0)
        if mask.any():
            yq[idx] = y[mask.argmax()]
        else:
            numer = np.sum((w / diffs) * y)
            denom = np.sum(w / diffs)
            yq[idx] = numer / denom
    return yq
