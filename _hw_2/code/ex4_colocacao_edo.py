
import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_diff_matrices

def solve(N):
    D, D2, x = cheb_diff_matrices(N)
    I = np.eye(N+1); X = np.diag(x)
    A = 20*(X@X)@D2 + X@D - I
    rhs = 5*(x**5) - 1
    A[0,:] = 0; A[0,0] = 1; rhs[0] = 2
    A[-1,:] = 0; A[-1,-1] = 1; rhs[-1] = 0
    y = np.linalg.solve(A, rhs)
    res = (20*(x**2)*(D2@y) + x*(D@y) - y - 5*(x**5) + 1)
    res[0] = 0; res[-1] = 0
    return x, y, res

def main():
    Ns = [8,16,32,64]
    plt.figure()
    for N in Ns:
        x, y, _ = solve(N)
        idx = np.argsort(x); plt.plot(x[idx], y[idx], label=f"N={N}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("y(x)")
    plt.title("Exercicio 4: Solucao por Colocacao de Chebyshev")
    plt.savefig("../figures/ex4_solutions.png", dpi=150, bbox_inches="tight")
    max_res = []
    for N in Ns:
        x, y, res = solve(N); max_res.append(np.max(np.abs(res)))
    plt.figure(); plt.semilogy(Ns, max_res, marker="o")
    plt.xlabel("N"); plt.ylabel("max |res| (log)")
    plt.title("Exercicio 4: Maximo do residuo vs N")
    plt.savefig("../figures/ex4_residuals.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
