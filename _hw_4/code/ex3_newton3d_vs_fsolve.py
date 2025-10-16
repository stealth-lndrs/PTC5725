#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercício 3 (PTC5725) — Sistema 3D:
    f1(x,y,z) = sin(xy) + exp(-xz) - 0.9 = 0
    f2(x,y,z) = z*sqrt(x^2 + y^2) - 6.7 = 0
    f3(x,y,z) = tan(y/x) + cos(z) + 3.2 = 0
Solução por Newton–Jacobian 3D (com e sem amortecimento) e alternativa com fsolve (SciPy, se disponível).
Saídas: figures/ex3_convergence_normF.pdf, figures/ex3_state_trajectory.pdf; tables/ex3_3d_newton_vs_fsolve_results.csv
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, importlib.util
from pathlib import Path

def F(v):
    x, y, z = v
    return np.array([
        np.sin(x*y) + np.exp(-x*z) - 0.9,
        z*np.sqrt(x**2 + y**2) - 6.7,
        np.tan(y/x) + np.cos(z) + 3.2
    ], dtype=float)

def J(v):
    x, y, z = v
    r = np.sqrt(x**2 + y**2)
    f1x = np.cos(x*y)*y - z*np.exp(-x*z)
    f1y = np.cos(x*y)*x
    f1z = -x*np.exp(-x*z)
    df2dx = 0.0 if r==0 else z*(x/r)
    df2dy = 0.0 if r==0 else z*(y/r)
    df2dz = r
    sec2 = 1.0/np.cos(y/x)**2 if x!=0 else np.inf
    f3x = sec2 * (-y/(x**2)) if x!=0 else np.inf
    f3y = sec2 * (1.0/x) if x!=0 else np.inf
    f3z = -np.sin(z)
    return np.array([[f1x, f1y, f1z],
                     [df2dx, df2dy, df2dz],
                     [f3x, f3y, f3z]], dtype=float)

def newton3d(x0, tol=1e-10, maxit=100, damping=True):
    x = np.array(x0, dtype=float)
    hist = []
    for it in range(1, maxit+1):
        f = F(x); nF = float(np.linalg.norm(f, ord=2))
        hist.append({"it": it, "x": x.copy(), "normF": nF})
        if nF < tol:
            return x, True, it, hist
        try:
            dx = np.linalg.solve(J(x), -f)
        except np.linalg.LinAlgError:
            return x, False, it-1, hist
        if damping:
            alpha = 1.0; fx = nF
            for _ in range(40):
                xt = x + alpha*dx
                if np.linalg.norm(F(xt)) < fx:
                    x = xt; break
                alpha *= 0.5
            else:
                x = x + dx
        else:
            x = x + dx
    return x, False, maxit, hist

def main():
    root = Path(__file__).resolve().parents[1]
    figdir, tabledir = root/'figures', root/'tables'
    figdir.mkdir(parents=True, exist_ok=True); tabledir.mkdir(parents=True, exist_ok=True)

    # Rodar alguns chutes
    inits = [(1.0,2.0,2.0), (0.8,1.8,2.0), (1.2,2.2,2.0), (1.0,2.0,2.2), (1.0,2.0,1.8)]
    rows = []
    for x0 in inits:
        sol, ok, it, _ = newton3d(x0, damping=True)
        rows.append({"method":"Newton-3D (damped)","x0":x0[0],"y0":x0[1],"z0":x0[2],
                     "x":float(sol[0]),"y":float(sol[1]),"z":float(sol[2]),"converged":ok,"iterations":it})
        sol2, ok2, it2, _ = newton3d(x0, damping=False)
        rows.append({"method":"Newton-3D (pure)","x0":x0[0],"y0":x0[1],"z0":x0[2],
                     "x":float(sol2[0]),"y":float(sol2[1]),"z":float(sol2[2]),"converged":ok2,"iterations":it2})

    # fsolve se disponível
    if importlib.util.find_spec("scipy") is not None:
        from scipy.optimize import fsolve
        sol, info, ier, msg = fsolve(lambda v: F(v), np.array([1.0,2.0,2.0]), fprime=lambda v: J(v), xtol=1e-12, full_output=True)
        rows.append({"method":"fsolve","x0":1.0,"y0":2.0,"z0":2.0,
                     "x":float(sol[0]),"y":float(sol[1]),"z":float(sol[2]),"converged":bool(ier==1),"iterations":int(info.get("nfev",0))})

    df = pd.DataFrame(rows)
    df.to_csv(tabledir/'ex3_3d_newton_vs_fsolve_results.csv', index=False)

    # Convergência
    _, _, _, hist = newton3d((1.0,2.0,2.0), damping=True)
    its = [h["it"] for h in hist]; norms = [h["normF"] for h in hist]
    plt.figure(); plt.semilogy(its, norms, '-o'); plt.xlabel('Iteração'); plt.ylabel('||F||_2'); plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Convergência do Newton 3D (amortecido)'); plt.savefig(figdir/'ex3_convergence_normF.pdf', bbox_inches='tight'); plt.close()

    xs = [h["x"][0] for h in hist]; ys = [h["x"][1] for h in hist]; zs = [h["x"][2] for h in hist]
    plt.figure(); plt.plot(its, xs, '-o', label='x_k'); plt.plot(its, ys, '-o', label='y_k'); plt.plot(its, zs, '-o', label='z_k')
    plt.xlabel('Iteração'); plt.ylabel('Valor'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
    plt.title('Trajetória das variáveis (Newton 3D amortecido)'); plt.savefig(figdir/'ex3_state_trajectory.pdf', bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    main()
