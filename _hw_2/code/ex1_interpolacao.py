
import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_lobatto_nodes, barycentric_weights, barycentric_eval

def main():
    f = lambda x: 1e3*np.sin(np.pi*x)
    n = 21
    x_nodes = cheb_lobatto_nodes(n)
    y_nodes = f(x_nodes)
    w = barycentric_weights(x_nodes)
    x_eval = np.linspace(-1,1,501)
    y_true = f(x_eval)
    y_bary = barycentric_eval(x_nodes, y_nodes, w, x_eval)
    V = np.vander(x_nodes, N=n+1, increasing=True)
    coeffs = np.linalg.solve(V, y_nodes)
    Xpow = np.vstack([x_eval**k for k in range(n+1)]).T
    y_lagr = Xpow @ coeffs
    plt.figure(); plt.plot(x_eval, y_true, label="f(x)")
    plt.plot(x_eval, y_bary, label="Baricentrica")
    plt.plot(x_eval, y_lagr, label="Lagrange")
    plt.scatter(x_nodes, y_nodes, s=12, label="Nodos Chebyshev")
    plt.title("Exercicio 1: Interpolacao (Lagrange vs Baricentrica)")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.savefig("../figures/ex1_interp.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.plot(x_eval, np.abs(y_bary - y_true), label="|baric - f|")
    plt.plot(x_eval, np.abs(y_lagr - y_true), label="|lagrange - f|")
    plt.yscale("log"); plt.xlabel("x"); plt.ylabel("erro abs."); plt.legend()
    plt.title("Erros de interpolacao (escala log)")
    plt.savefig("../figures/ex1_errors.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
