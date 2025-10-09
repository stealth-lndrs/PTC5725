import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_basis_matrix, cheb_diff_matrices

def cheb_nodes_standard(n):
    """Nós de Chebyshev–Lobatto: x0=+1 ... xn=-1."""
    return np.cos(np.pi * np.arange(0, n+1) / n)

def cheb_basis_matrix_standard(n):
    """Matriz de base de Chebyshev T_j(x_i) para x padrão (x0=+1 → xn=-1)."""
    x = cheb_nodes_standard(n)
    theta = np.arccos(x)
    B = np.zeros((n+1, n+1))
    for j in range(n+1):
        B[:, j] = np.cos(j * theta)
    return x, B

def cheb_diff_matrices_standard(n):
    """Matrizes diferenciais de Chebyshev D e D² (Trefethen, 2000)."""
    if n == 0:
        return np.array([[0.0]]), np.array([[0.0]]), np.array([1.0])
    x = cheb_nodes_standard(n)
    c = np.ones(n+1)
    c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0)**np.arange(n+1))
    X = np.tile(x, (n+1, 1))
    dX = X - X.T + np.eye(n+1)
    D = (np.outer(c, 1/c)) / dX
    D = D - np.diag(np.sum(D, axis=1))
    D2 = D @ D
    return D, D2, x

def solve_series_cheb_standard(N=8):
    """
    Resolve numericamente:
        20x²y'' + xy' - y - 5x⁵ + 1 = 0,   y(-1)=2, y(1)=0
    via série de Chebyshev truncada de grau N.
    """
    x, B = cheb_basis_matrix_standard(N)
    D, D2, _ = cheb_diff_matrices_standard(N)

    X = np.diag(x)
    Ly = 20*(X @ X) @ D2 + X @ D - np.eye(N+1)
    f = 5*x**5 - 1
    A = Ly @ B

    # Impor BCs nas linhas correspondentes
    A[0, :]  = B[0, :]   # x=+1 → y(1)=0
    A[-1, :] = B[-1, :]  # x=-1 → y(-1)=2
    f[0] = 0.0
    f[-1] = 2.0

    # Resolver para os coeficientes espectrais
    a = np.linalg.solve(A, f)
    y = B @ a
    return x, y

def main():
    plt.figure(figsize=(7,4))
    for N in [7, 8]:
        x, y = solve_series_cheb_standard(N)
        idx = np.argsort(x)
        plt.plot(x[idx], y[idx], 'o-', label=f"Série grau {N}")
    # Marcar as condições de contorno
    plt.scatter([1, -1], [0, 2], c='k', marker='x', zorder=5, label='BC esperada')
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Ex. 4 – Solução numérica por série de Chebyshev (graus 7 e 8)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/ex4_serie_cheb_numerica_BCfixed_v2.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
