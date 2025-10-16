#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercício 1 (PTC5725) — BVP: y'' = exp(y), y(-1)=y(1)=1
Solução numérica por colocalização de Chebyshev + Newton–Raphson.

Funcionalidades:
- Gera matriz diferencial de Chebyshev (Trefethen);
- Resolve o sistema não linear via Newton com imposição forte de Dirichlet;
- Salva amostras (CSV), resumo (JSON) e figuras (PDF);
- Executa estudo de refinamento simples entre duas grades (N1=48, N2=96).

Como usar:
$ python code/ex1_cheb_newton.py
Os arquivos de saída serão colocados em ../tables e ../figures relativos a este script.
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilidades de Chebyshev
# ---------------------------------------------------------------------
def cheb(N: int):
    """
    Matriz de diferenciação de Chebyshev (nós de Chebyshev–Lobatto) e nós x.
    Implementação baseada em "Spectral Methods in MATLAB", Trefethen (1996).

    Parâmetro
    ---------
    N : int
        Ordem (numero de subintervalos polinomiais). O número de nós será N+1.

    Retorno
    -------
    D : ndarray (N+1, N+1)
        Matriz de diferenciação de primeira ordem.
    x : ndarray (N+1,)
        Nós de Chebyshev-Lobatto em [-1, 1].
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])

    x = np.cos(np.pi * np.arange(N + 1) / N)  # nós
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T

    # Fórmula fechada com "truque" da identidade (diagonal tratada separadamente)
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def cheb_coeffs(y: np.ndarray) -> np.ndarray:
    """
    Coeficientes c_k da expansão de Chebyshev de y(x) amostrado nos nós de Chebyshev-Lobatto.
    Usamos uma DCT-I simplificada (definição explícita via cossenos).

    Retorna c (N+1,), onde y(x) ≈ sum_{k=0}^N c_k T_k(x).
    """
    N = len(y) - 1
    theta = np.pi * np.arange(N + 1) / N
    c = np.zeros(N + 1)
    for m in range(N + 1):
        c[m] = (2.0 / N) * np.sum(y * np.cos(m * theta))
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


# ---------------------------------------------------------------------
# Solver Newton–Raphson para o BVP y'' = exp(y), y(±1)=1
# ---------------------------------------------------------------------
def solve_bvp_cheb_newton(N: int = 96, tol: float = 1e-12, maxit: int = 50):
    """
    Resolve o BVP via colocalização de Chebyshev (D,D^2) + Newton.

    Retorna
    -------
    x : nós de Chebyshev-Lobatto
    y : solução aproximada em x
    R : resíduo R = y'' - exp(y) em x (nas bordas setado para 0)
    Rp: derivada do resíduo R' ≈ D*R
    info : dicionário com métricas de convergência
    """
    D, x = cheb(N)
    D2 = D @ D

    # Chute inicial: constante 1 (satisfaz Dirichlet)
    y = np.ones(N + 1)
    it = 0
    for it in range(1, maxit + 1):
        # Resíduo não linear F(y) = D2 y - exp(y)
        F = D2 @ y - np.exp(y)

        # Impõe Dirichlet fortemente nas linhas de fronteira
        F[0] = y[0] - 1.0
        F[-1] = y[-1] - 1.0

        # Jacobiano J = D2 - diag(exp(y)), com linhas de fronteira como identidade
        J = D2 - np.diag(np.exp(y))
        J[0, :] = 0.0
        J[0, 0] = 1.0
        J[-1, :] = 0.0
        J[-1, -1] = 1.0

        # Passo de Newton
        dy = np.linalg.solve(J, -F)
        y = y + dy

        if np.linalg.norm(dy, ord=np.inf) < tol:
            break

    # Avalia resíduo final e sua derivada
    R = D2 @ y - np.exp(y)
    R[0] = 0.0
    R[-1] = 0.0
    Rp = D @ R

    info = {
        "N": N,
        "iterations": it,
        "step_inf_norm": float(np.linalg.norm(dy, ord=np.inf)),
        "residual_inf_norm": float(np.linalg.norm(R[1:-1], ord=np.inf)),
        "residual_L2_norm": float(np.linalg.norm(R[1:-1]) / math.sqrt(N - 1)),
    }
    return x, y, R, Rp, info


# ---------------------------------------------------------------------
# Interpolação bariocêntrica (para estudo de refinamento)
# ---------------------------------------------------------------------
def barycentric_weights_cheb(N: int) -> np.ndarray:
    """
    Pesos bariocêntricos para nós de Chebyshev–Lobatto.
    """
    w = np.ones(N + 1)
    w[0] = 0.5
    w[-1] = 0.5
    w = w * ((-1.0) ** np.arange(N + 1))
    return w


def barycentric_interpolate(xnodes, w, fvals, xeval):
    """
    Interpolação de Lagrange na forma bariocêntrica (avaliando em xeval).
    """
    xnodes = np.asarray(xnodes)
    w = np.asarray(w)
    fvals = np.asarray(fvals)
    xeval = np.asarray(xeval)
    out = np.empty_like(xeval, dtype=float)
    for i, xv in enumerate(xeval):
        diff = xv - xnodes
        j = np.where(np.abs(diff) < 1e-14)[0]
        if j.size > 0:
            out[i] = fvals[j[0]]
        else:
            tmp = w / diff
            out[i] = np.sum(tmp * fvals) / np.sum(tmp)
    return out


# ---------------------------------------------------------------------
# Rotina principal (gera saídas de tabelas e figuras)
# ---------------------------------------------------------------------
def main():
    # Pastas de saída relativas a este arquivo
    here = Path(__file__).resolve().parent
    figdir = (here.parent / "figures")
    tabledir = (here.parent / "tables")
    figdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    # 1) Resolver com N=96
    x, y, R, Rp, info = solve_bvp_cheb_newton(N=96)

    # 2) Salvar tabelas
    df = pd.DataFrame({"x": x, "y": y, "R": R, "Rprime": Rp})
    df.to_csv(tabledir / "ex1_solution_samples.csv", index=False)
    with open(tabledir / "ex1_summary.json", "w") as f:
        json.dump(info, f, indent=2)

    # 3) Figuras
    plt.figure()
    plt.plot(x, y)
    plt.scatter(x, y, s=12)
    plt.xlabel("x"); plt.ylabel("y(x)"); plt.title("Exercício 1: y(x) com pontos")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figdir / "ex1_solution_with_points.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(x, y, s=16)
    plt.xlabel("x"); plt.ylabel("y nos nós"); plt.title("Exercício 1: nós de Chebyshev")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figdir / "ex1_points_only.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(x, R)
    plt.xlabel("x"); plt.ylabel("R(x)"); plt.title("Exercício 1: Resíduo")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figdir / "ex1_residual.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(x, Rp)
    plt.xlabel("x"); plt.ylabel("R'(x)"); plt.title("Exercício 1: Derivada do Resíduo")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figdir / "ex1_residual_prime.pdf", bbox_inches="tight")
    plt.close()

    c = cheb_coeffs(y)
    plt.figure()
    plt.semilogy(np.arange(len(c)), np.abs(c))
    plt.xlabel("índice k"); plt.ylabel("|c_k|"); plt.title("Exercício 1: decaimento espectral")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figdir / "ex1_coeff_decay.pdf", bbox_inches="tight")
    plt.close()

    # 4) Estudo de refinamento simples (N1=48 vs N2=96) em malha fina M=200
    N1, N2 = 48, 96
    x1, y1, *_ = solve_bvp_cheb_newton(N=N1)
    w1 = barycentric_weights_cheb(N1)
    w2 = barycentric_weights_cheb(N2)
    xfine = np.cos(np.pi * np.arange(201) / 200)

    y1f = barycentric_interpolate(x1, w1, y1, xfine)
    y2f = barycentric_interpolate(x, w2, y, xfine)
    diff = np.abs(y2f - y1f)
    df_ref = pd.DataFrame([{
        "N1": N1, "N2": N2,
        "Linf_on_fine": float(np.max(diff)),
        "L2_on_fine": float(np.linalg.norm(diff) / math.sqrt(len(diff)))
    }])
    df_ref.to_csv(tabledir / "ex1_refinement_study.csv", index=False)

    print("Concluído. Saídas gravadas em:")
    print(" -", figdir)
    print(" -", tabledir)


if __name__ == "__main__":
    main()
