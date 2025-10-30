#!/usr/bin/env python3
# Exercício 1 — EDO não linear (Chebyshev + Newton) com BCs corretas
# EDO: u'' - u^2 - 2 + (x^2 - 5x + 6)^2 = 0,  u(-1)=12, u(1)=2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cheb(N: int):
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N+1))
    X = np.tile(x, (N+1, 1))
    dX = X - X.T
    D = (np.outer(c, 1.0/c)) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def build_system(u, D2, x):
    F = D2 @ u - (u**2) - 2.0 + (x**2 - 5.0*x + 6.0)**2
    J = D2 - np.diag(2.0*u)
    return F, J

def apply_dirichlet_correct(F, J, u, x, uL=12.0, uR=2.0):
    N = len(u)-1
    # x[0]=+1, x[N]=-1 na ordenação padrão de Chebyshev; imponha BCs corretamente:
    i_right = 0 if x[0] > x[-1] else N   # índice de x=+1
    i_left  = N - i_right                 # índice de x=-1
    J[i_left,:]  = 0.0; J[i_left,i_left]  = 1.0; F[i_left]  = u[i_left]  - uL
    J[i_right,:] = 0.0; J[i_right,i_right]= 1.0; F[i_right] = u[i_right] - uR
    return F, J

def solve_ex1(N=64, tol=1e-12, maxit=50):
    D, x = cheb(N); D2 = D @ D
    # chute inicial coerente com as BCs
    u = np.interp(x, [-1.0, 1.0], [12.0, 2.0])
    for it in range(maxit):
        F, J = build_system(u, D2, x)
        F, J = apply_dirichlet_correct(F, J, u, x)
        du = np.linalg.solve(J, -F)
        u += du
        if np.linalg.norm(du, np.inf) < tol:
            break
    return x, u, it

def make_figures(x, u, outdir="figures"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    u_exact = x**2 - 5.0*x + 6.0
    err = u - u_exact

    plt.figure()
    plt.plot(x, u, '-', label='u (numérica)')
    plt.plot(x, u_exact, '--', label='u (analítica)')
    plt.scatter(x, u, s=10)
    plt.legend(); plt.xlabel('x'); plt.ylabel('u')
    plt.title('Ex1: Solução numérica vs analítica (corrigido BC)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out/'ex1c_solution_compare.pdf', bbox_inches='tight'); plt.close()

    D, _ = cheb(len(x)-1); D2 = D @ D
    F = D2 @ u - (u**2) - 2.0 + (x**2 - 5.0*x + 6.0)**2
    F[0] = 0.0; F[-1] = 0.0
    plt.figure(); plt.plot(x, F, '-')
    plt.xlabel('x'); plt.ylabel('F(u)')
    plt.title('Ex1: Resíduo F(u) nos nós (corrigido)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out/'ex1c_residual.pdf', bbox_inches='tight'); plt.close()

    plt.figure(); plt.plot(x, err, '-')
    plt.xlabel('x'); plt.ylabel('u - u_exato')
    plt.title('Ex1: Erro ponto a ponto (corrigido)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out/'ex1c_pointwise_error.pdf', bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    x, u, it = solve_ex1(64)
    make_figures(x, u, "figures")
