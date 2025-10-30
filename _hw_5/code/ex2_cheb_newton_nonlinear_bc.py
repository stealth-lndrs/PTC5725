#!/usr/bin/env python3
# Exercício 2 — y''*y + (y')^2 - 2 e^{-2x} = 0,  y(-1)=e, y(1)=e^{-1}
# Método: Colocação de Chebyshev + Newton com amortecimento (backtracking)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, pandas as pd

def cheb(N: int):
    if N == 0: return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1); c[0] = 2.0; c[-1] = 2.0
    c = c*((-1.0)**np.arange(N+1))
    X = np.tile(x, (N+1, 1)); dX = X - X.T
    D = (np.outer(c, 1.0/c)) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def residual_and_jacobian(y, D, D2, x):
    yp  = D @ y
    ypp = D2 @ y
    F = ypp * y + yp**2 - 2.0*np.exp(-2.0*x)
    J = np.diag(y) @ D2 + np.diag(ypp) + 2.0*np.diag(yp) @ D
    return F, J

def apply_dirichlet(F, J, y, x, yL=np.e, yR=np.e**(-1)):
    N = len(y)-1
    i_right = 0 if x[0] > x[-1] else N   # x=+1
    i_left  = N - i_right                # x=-1
    J[i_left,:]  = 0.0; J[i_left,i_left]   = 1.0; F[i_left]  = y[i_left]  - yL
    J[i_right,:] = 0.0; J[i_right,i_right] = 1.0; F[i_right] = y[i_right] - yR
    return F, J

def newton_cheb_ex2(N=64, tol=1e-12, maxit=50, damping=True):
    D, x = cheb(N); D2 = D @ D
    y = np.ones(N+1, dtype=float)   # chute inicial
    hist = []
    for it in range(1, maxit+1):
        F, J = residual_and_jacobian(y, D, D2, x)
        nF_raw = float(np.linalg.norm(F, ord=2))
        F, J = apply_dirichlet(F, J, y, x)
        dy = np.linalg.solve(J, -F)
        if damping:
            alpha = 1.0; fy = nF_raw
            for _ in range(40):
                yt = y + alpha*dy
                Ft, _ = residual_and_jacobian(yt, D, D2, x)
                nFt = float(np.linalg.norm(Ft, ord=2))
                if nFt < fy: y = yt; break
                alpha *= 0.5
            else:
                y = y + dy
        else:
            y = y + dy
        hist.append({"it": it, "normF": nF_raw, "max_step": float(np.max(np.abs(dy)))})
        if np.linalg.norm(dy, ord=np.inf) < tol: 
            return x, y, True, hist
    return x, y, False, hist

def main():
    out_root = Path("ex2_output"); figdir = out_root/"figures"; tabledir = out_root/"tables"
    figdir.mkdir(parents=True, exist_ok=True); tabledir.mkdir(parents=True, exist_ok=True)
    x, y, ok, hist = newton_cheb_ex2()
    pd.DataFrame(hist).to_csv(tabledir/"ex2_convergence.csv", index=False)
    with open(tabledir/"ex2_summary.json","w") as f:
        json.dump({"N":64,"converged":bool(ok),"iterations":hist[-1]["it"] if hist else 0,
                   "final_normF":hist[-1]["normF"] if hist else None}, f, indent=2)

    # figures
    D, _ = cheb(64); D2 = D @ D
    y_exact = np.exp(-x)
    F, _ = residual_and_jacobian(y, D, D2, x); F[0]=0.0; F[-1]=0.0
    err = y - y_exact

    plt.figure(); plt.plot(x,y,'-',label='y (numérica)')
    plt.plot(x,y_exact,'--',label='y (analítica)'); plt.scatter(x,y,s=10)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Ex2: y(x) numérica vs analítica')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(figdir/'ex2_solution_compare.pdf', bbox_inches='tight'); plt.close()

    plt.figure(); plt.plot(x,F,'-'); plt.xlabel('x'); plt.ylabel('F(y)')
    plt.title('Ex2: Resíduo F(y) nos nós (interior)'); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(figdir/'ex2_residual.pdf', bbox_inches='tight'); plt.close()

    plt.figure(); plt.plot(x,err,'-'); plt.xlabel('x'); plt.ylabel('y - y_exato')
    plt.title('Ex2: Erro ponto a ponto'); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(figdir/'ex2_pointwise_error.pdf', bbox_inches='tight'); plt.close()

    its = [h["it"] for h in hist]; nF = [h["normF"] for h in hist]
    plt.figure(); plt.semilogy(its,nF,'-o'); plt.xlabel('iteração'); plt.ylabel(r'||F||_2 (pré-BC)')
    plt.title('Ex2: Convergência do Newton (amortecido)'); plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(figdir/'ex2_convergence_normF.pdf', bbox_inches='tight'); plt.close()

if __name__=="__main__":
    main()
