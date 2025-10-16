#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercício 2 (PTC5725) — Sistema não linear:
    f1(x,y) = x^3 + y - 1 = 0
    f2(x,y) = y^3 - x + 1 = 0
Métodos: Newton–Jacobian (com e sem amortecimento) e fsolve (SciPy).
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, importlib.util
from pathlib import Path

def F(v):
    x, y = v
    return np.array([x**3 + y - 1.0, y**3 - x + 1.0])

def J(v):
    x, y = v
    return np.array([[3*x**2, 1.0], [-1.0, 3*y**2]])

def newton_jacobian(x0, tol=1e-12, maxit=50, damping=False):
    x = np.array(x0, dtype=float)
    for it in range(maxit):
        f = F(x); normF = np.linalg.norm(f)
        if normF < tol:
            return x, True, it
        dx = np.linalg.solve(J(x), -f)
        if damping:
            alpha = 1.0
            while alpha > 1e-6:
                if np.linalg.norm(F(x + alpha*dx)) < normF:
                    x = x + alpha*dx; break
                alpha *= 0.5
            else:
                x = x + dx
        else:
            x = x + dx
    return x, False, maxit

def main():
    root = Path(__file__).resolve().parents[1]
    figdir, tabledir = root/'figures', root/'tables'
    figdir.mkdir(exist_ok=True, parents=True); tabledir.mkdir(exist_ok=True, parents=True)
    inits = [(0.5,0.5),(1.5,0.5),(-0.5,0.5),(2.0,-1.0),(-2.0,2.0)]
    rows = []
    for x0 in inits:
        sol, ok, it = newton_jacobian(x0, damping=False)
        rows.append({'method':'Newton','x0':x0[0],'y0':x0[1],'x':sol[0],'y':sol[1],'ok':ok,'it':it})
        sol, ok, it = newton_jacobian(x0, damping=True)
        rows.append({'method':'Newton (damped)','x0':x0[0],'y0':x0[1],'x':sol[0],'y':sol[1],'ok':ok,'it':it})
    if importlib.util.find_spec('scipy'):
        from scipy.optimize import fsolve
        sol,info,ier,msg = fsolve(F,[0.5,0.5],fprime=J,xtol=1e-12,full_output=True)
        rows.append({'method':'fsolve','x0':0.5,'y0':0.5,'x':sol[0],'y':sol[1],'ok':ier==1,'it':info.get('nfev',0)})
    df = pd.DataFrame(rows); df.to_csv(tabledir/'ex2_newton_vs_fsolve_results.csv',index=False)
    xx=np.linspace(-3,3,300);yy=np.linspace(-3,3,300);X,Y=np.meshgrid(xx,yy)
    F1=X**3+Y-1;F2=Y**3-X+1
    plt.contour(X,Y,F1,[0]);plt.contour(X,Y,F2,[0],linestyles='--')
    for x0 in inits:
        s,_,_=newton_jacobian(x0,True);plt.plot(s[0],s[1],'o')
    plt.plot([1],[0],'r*',ms=10);plt.savefig(figdir/'ex2_contours.pdf');plt.close()
if __name__=='__main__': main()
