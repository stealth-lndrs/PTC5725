import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm
import pandas as pd

# ============================================================
# Funções utilitárias: Matrizes de diferenciação de Chebyshev
# ============================================================

def cheb_D_matrices(N):
    """Retorna nós de Chebyshev e matrizes de diferenciação D e D2."""
    k = np.arange(0, N+1)
    x = np.cos(np.pi * k / N)[::-1]  # nós em ordem crescente
    X = np.tile(x, (N+1,1))
    dX = X - X.T
    c = np.ones(N+1)
    c[0] = 2; c[-1] = 2
    c = c * ((-1)**np.arange(N+1))
    C = np.tile(c, (N+1,1))
    D = (C.T / C) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    D2 = D @ D
    return x, D, D2

# ============================================================
# Solver de Newton para u'' = e^u, u(±1)=1
# ============================================================

def newton_solve_bvp(N=64, tol=1e-12, maxit=30):
    x, D, D2 = cheb_D_matrices(N)
    idx_all = np.arange(N+1)
    idx_int = idx_all[1:-1]
    u = np.ones(N+1)
    u[0] = 1.0; u[-1] = 1.0
    hist = []
    for it in range(1, maxit+1):
        R = D2 @ u - np.exp(u)
        R[0] = 0.0; R[-1] = 0.0
        r = R[idx_int]
        resn = norm(r, ord=np.inf)
        hist.append(resn)
        if resn < tol:
            break
        J = D2[np.ix_(idx_int, idx_int)] - np.diag(np.exp(u[idx_int]))
        du_int = solve(J, -r)
        u[idx_int] += du_int
    R = D2 @ u - np.exp(u)
    R[0] = 0.0; R[-1] = 0.0
    return x, u, R, np.array(hist), D2

# ============================================================
# Execução principal
# ============================================================

x, u, R, hist, D2 = newton_solve_bvp(N=64, tol=1e-12, maxit=50)

# ============================================================
# Solução analítica aproximada: u = log(a^2/(1+cos(a x)))
# ============================================================

def f_a(a): return a*a - np.e*(1.0 + np.cos(a))
def df_a(a): return 2*a + np.e*np.sin(a)

def find_a_newton(a0=2.0, tol=1e-14, maxit=100):
    a = a0
    for _ in range(maxit):
        fa = f_a(a)
        dfa = df_a(a)
        if dfa == 0: break
        step = fa / dfa
        a -= step
        if abs(step) < tol:
            break
    return a

a_star = find_a_newton(2.0)

def u_analytic(x, a):
    return np.log(a*a / (1.0 + np.cos(a*x)))

u_an = u_analytic(x, a_star)

# Erros
mask_int = (x > -1+1e-14) & (x < 1-1e-14)
err = u - u_an
err_inf = np.max(np.abs(err[mask_int]))
err_L2 = np.sqrt(np.trapz(err[mask_int]**2, x[mask_int]))

# ============================================================
# Gráficos
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(x, u, 'o-', label="u (numérico)", markersize=5)
plt.plot(x, u_an, 's--', label="u_an(a*)", markersize=5)
plt.xlabel("x"); plt.ylabel("u(x)")
plt.title("Comparação entre solução numérica e analítica")
plt.legend(); plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("u_solution_highlighted.png", dpi=200)
plt.show()

plt.figure()
plt.plot(x, R)
plt.xlabel("x"); plt.ylabel("R(x)")
plt.title("Resíduo R(x) = u'' - e^u")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("residual.png", dpi=160)
plt.show()

plt.figure()
plt.semilogy(np.arange(1, len(hist)+1), hist, marker='o')
plt.xlabel("Iteração de Newton"); plt.ylabel("||R_int||_inf")
plt.title("Convergência de Newton")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("newton_convergence.png", dpi=160)
plt.show()

# ============================================================
# Tabela de resultados
# ============================================================

df = pd.DataFrame({
    "N+1 (nós)": [len(x)],
    "Iterações Newton": [len(hist)],
    "||R||_inf (final)": [float(np.max(np.abs(R)))],
    "a* (da BC analítica)": [float(a_star)],
    "Erro_inf(u - u_an)": [float(err_inf)],
    "Erro_L2(u - u_an)": [float(err_L2)],
})
print(df)
