
import numpy as np

def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    x_nodes = np.array(x_nodes, dtype=float)
    y_nodes = np.array(y_nodes, dtype=float)
    x_eval  = np.array(x_eval,  dtype=float)
    n = len(x_nodes)
    y_eval = np.zeros_like(x_eval, dtype=float)
    for j in range(n):
        lj = np.ones_like(x_eval, dtype=float)
        for m in range(n):
            if m != j:
                lj *= (x_eval - x_nodes[m]) / (x_nodes[j] - x_nodes[m])
        y_eval += y_nodes[j] * lj
    return y_eval
