
import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_lobatto_nodes

def main():
    n = 16
    x = cheb_lobatto_nodes(n, neg_flip=True)
    theta = np.arccos(x)
    B = np.zeros((n+1, n+1))
    for j in range(n+1):
        B[:, j] = np.cos(j*theta)
    np.savetxt("ex2_nodes_x.csv", x, delimiter=",")
    np.savetxt("ex2_basis_B.csv", B, delimiter=",")
    plt.figure(); plt.imshow(B, aspect="auto", origin="lower"); plt.colorbar()
    plt.title("Exercicio 2: Matriz de Base Chebyshev B (n=16)")
    plt.xlabel("j (grau)"); plt.ylabel("i (nodo)")
    plt.savefig("../figures/ex2_basis_heatmap.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
