"""
Legendre orthogonality verification (PCS5029 - Aula 06 - Exercício 5)
Requirements: numpy, matplotlib
Notes:
- Quadrature uses numpy.polynomial.legendre.leggauss(N) with N=400.
- Saves figures under /figures and tables under /tables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre, leggauss
import pandas as pd

# Parameters
N_QUAD = 400
MAX_N_FOR_MATRIX = 10
MAX_N_FOR_PLOT = 5
BASE_DIR = "/mnt/data"
FIG_DIR = os.path.join(BASE_DIR, "figures")
TAB_DIR = os.path.join(BASE_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def legendre_basis(n: int) -> Legendre:
    return Legendre.basis(n)

def inner_product_matrix(n_max: int, Nq: int) -> np.ndarray:
    xi, wi = leggauss(Nq)
    Pvals = np.zeros((n_max+1, xi.size))
    for n in range(n_max+1):
        Pvals[n, :] = legendre_basis(n)(xi)
    G = np.zeros((n_max+1, n_max+1))
    for m in range(n_max+1):
        for n in range(n_max+1):
            G[m, n] = np.sum(wi * Pvals[m, :] * Pvals[n, :])
    return G

def main():
    # Compute Gram matrix
    G = inner_product_matrix(MAX_N_FOR_MATRIX, N_QUAD)
    theoretical_diag = np.array([2.0/(2*n+1) for n in range(MAX_N_FOR_MATRIX+1)])

    # Save tables
    G_df = pd.DataFrame(G, index=[f"P{m}" for m in range(MAX_N_FOR_MATRIX+1)],
                           columns=[f"P{n}" for n in range(MAX_N_FOR_MATRIX+1)])
    G_df.to_csv(os.path.join(TAB_DIR, "legendre_orthogonality_matrix.csv"), float_format="%.12e")

    G_err_df = G_df.copy()
    for n in range(MAX_N_FOR_MATRIX+1):
        for m in range(MAX_N_FOR_MATRIX+1):
            target = 0.0 if m != n else theoretical_diag[n]
            G_err_df.iloc[m, n] = G_df.iloc[m, n] - target
    G_err_df.to_csv(os.path.join(TAB_DIR, "legendre_orthogonality_error_matrix.csv"), float_format="%.12e")

    pd.DataFrame({
        "n": np.arange(MAX_N_FOR_MATRIX+1),
        "theoretical_2_over_2n_plus_1": theoretical_diag,
        "numerical_diag": np.diag(G),
        "abs_error": np.abs(np.diag(G) - theoretical_diag)
    }).to_csv(os.path.join(TAB_DIR, "legendre_norms_comparison.csv"),
               index=False, float_format="%.12e")

    # Plot P0..P5
    x_plot = np.linspace(-1, 1, 1000)
    plt.figure()
    for n in range(MAX_N_FOR_PLOT+1):
        plt.plot(x_plot, legendre_basis(n)(x_plot), label=fr"$P_{5}(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$P_n(x)$")
    plt.title("Legendre Polynomials $P_0$ to $P_5$")
    plt.legend(loc="best", ncol=2, frameon=True)
    plt.grid(True, which="both", linestyle=":")
    plt.savefig(os.path.join(FIG_DIR, "legendre_polynomials.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Heatmap of G
    plt.figure()
    im = plt.imshow(G, origin="lower", extent=[-0.5, MAX_N_FOR_MATRIX+0.5, -0.5, MAX_N_FOR_MATRIX+0.5])
    plt.colorbar(im, label=r"$\int_{-1}^{1} P_m(x)P_n(x)\,dx$")
    plt.xticks(range(0, MAX_N_FOR_MATRIX+1), [f"P{n}" for n in range(MAX_N_FOR_MATRIX+1)], rotation=45)
    plt.yticks(range(0, MAX_N_FOR_MATRIX+1), [f"P{m}" for m in range(MAX_N_FOR_MATRIX+1)])
    plt.title("Orthogonality Matrix for Legendre Polynomials")
    plt.savefig(os.path.join(FIG_DIR, "legendre_orthogonality_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
