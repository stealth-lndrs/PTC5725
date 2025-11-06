"""
Aula 07 | Exercício 2 – Transformações entre coeficientes e valores nodais (Legendre)
--------------------------------------------------------------------------
Gera B_L, B_L^{-1}, verifica identidades e salva figuras/tabelas.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre

def lgl_nodes_weights(N: int):
    if N < 1:
        raise ValueError("N must be >= 1 for Gauss-Lobatto.")
    Pn = Legendre.basis(N)
    dPn = Pn.deriv()
    interior = np.sort(dPn.roots())
    x = np.empty(N + 1)
    x[0] = -1.0
    x[-1] = 1.0
    if N > 1:
        x[1:-1] = interior
    Px = Pn(x)
    w = 2.0 / (N * (N + 1) * (Px ** 2))
    return x, w

def build_BL_matrix(N: int):
    x, _ = lgl_nodes_weights(N)
    BL = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        Pk = Legendre.basis(k)
        BL[:, k] = Pk(x)
    return BL

def build_BL_inverse(N: int):
    x, w = lgl_nodes_weights(N)
    BL = build_BL_matrix(N)
    W = np.diag(w)
    G = BL.T @ W @ BL
    BLinv = np.linalg.solve(G, BL.T @ W)
    return BLinv

def verify_identity(BL: np.ndarray, BLinv: np.ndarray):
    I = np.eye(BL.shape[0])
    err_left = np.max(np.abs(I - BL @ BLinv))
    err_right = np.max(np.abs(I - BLinv @ BL))
    return err_left, err_right

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    N_fig = 24
    BL = build_BL_matrix(N_fig)
    BLinv = build_BL_inverse(N_fig)
    eL, eR = verify_identity(BL, BLinv)

    # Heatmap BL
    plt.figure(figsize=(6, 4.5))
    plt.imshow(BL, aspect="auto")
    plt.colorbar()
    plt.title(f"B_L matrix (N={N_fig})")
    plt.xlabel("k (Legendre degree)")
    plt.ylabel("i (LGL node index)")
    plt.tight_layout()
    plt.savefig("figures/BL_matrix_heatmap.png", dpi=200)
    plt.close()

    # Heatmap error
    I = np.eye(N_fig + 1)
    err_mat = np.abs(I - BL @ BLinv)
    plt.figure(figsize=(6, 4.5))
    plt.imshow(err_mat, aspect="auto")
    plt.colorbar()
    plt.title(r"$|I - B_L B_L^{-1}|$" + f" (N={{N_fig}})".format(N_fig=N_fig))
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig("figures/BL_inverse_error.png", dpi=200)
    plt.close()

    # Table for multiple N
    Ns = [4, 8, 12, 16, 24, 32, 40, 48, 64]
    rows = []
    for N in Ns:
        BLN = build_BL_matrix(N)
        BLi = build_BL_inverse(N)
        eL, eR = verify_identity(BLN, BLi)
        rows.append({"N": N, "||I - BL BLinv||_inf": eL, "||I - BLinv BL||_inf": eR})
    df = pd.DataFrame(rows)
    df.to_csv("tables/legendre_matrix_errors.csv", index=False)
    with open("tables/legendre_matrix_errors.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.2e}"))

if __name__ == "__main__":
    main()
