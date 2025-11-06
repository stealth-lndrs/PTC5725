# Aula 07 – Tarefa 1 (Quadraturas Espectrais) – Python
# Gera tabelas e figuras de convergência para 4 integrais usando
# Gauss-Legendre, Gauss-Lobatto, Gauss-Radau, Fejér I e Clenshaw-Curtis.

import os, math
from typing import Callable, Tuple
import numpy as np
import mpmath as mp
import pandas as pd
import matplotlib.pyplot as plt

def map_to_interval(z, w, a, b):
    z_scaled = 0.5*((b-a)*z + (b+a))
    w_scaled = 0.5*(b-a)*w
    return z_scaled, w_scaled

def gauss_legendre(n):
    return np.polynomial.legendre.leggauss(n)

def legendre_P(n, x):
    c = np.zeros(n+1); c[-1] = 1.0
    Pn = np.polynomial.legendre.Legendre(c)
    return Pn(x)

def gauss_lobatto(n):
    if n < 2:
        raise ValueError("Lobatto requer n >= 2")
    z = np.empty(n); z[0], z[-1] = -1.0, 1.0
    if n > 2:
        c = np.zeros(n); c[-1] = 1.0
        dP = np.polynomial.legendre.Legendre(c).deriv()
        z[1:-1] = np.sort(dP.roots())
    Pn1 = legendre_P(n-1, z)
    w = 2.0/((n-1)*n*(Pn1**2))
    return z, w

def gauss_radau_left(n):
    if n < 1:
        raise ValueError("Radau requer n >= 1")
    c_n1 = np.zeros(n); c_n1[-1] = 1.0
    Pn1 = np.polynomial.legendre.Legendre(c_n1)
    c_n = np.zeros(n+1); c_n[-1] = 1.0
    Pn = np.polynomial.legendre.Legendre(c_n)
    poly_sum = Pn1 + Pn
    roots = np.sort(poly_sum.roots())
    z = np.empty(n); z[0] = -1.0; z[1:] = roots
    w = (1.0 - z) / (n**2 * (Pn1(z)**2))
    return z, w

def fejer_type1(n):
    k = np.arange(1, n+1)
    theta = (2*k - 1)*np.pi/(2*n)
    z = np.cos(theta)
    w = np.zeros(n, dtype=float)
    J = int((n-1)//2)
    for idx, th in enumerate(theta):
        s = 0.0
        for j in range(1, J+1):
            s += np.cos(2*j*th)/(4*j*j - 1)
        w[idx] = (2.0/n)*(1 - 2*s)
    return z, w

def clenshaw_curtis(n):
    N = n-1
    k = np.arange(0, N+1)
    theta = np.pi*k/N
    z = -np.cos(theta)
    w = np.zeros(n, dtype=float)
    if N == 0:
        w[0] = 2.0; return z, w
    if N % 2 == 0:
        w[0] = 1.0/(N**2 - 1); w[-1] = w[0]
        v = np.ones(n-2)
        for k_ in range(1, N//2):
            v -= 2*np.cos(2*k_*theta[1:-1])/(4*k_*k_ - 1)
        v -= np.cos(N*theta[1:-1])/(N**2 - 1)
        w[1:-1] = 2*v/N
    else:
        w[0] = 1.0/N**2; w[-1] = w[0]
        v = np.ones(n-2)
        for k_ in range(1, (N+1)//2):
            v -= 2*np.cos(2*k_*theta[1:-1])/(4*k_*k_ - 1)
        w[1:-1] = 2*v/N
    return z, w

def integrate_rule(f, a, b, rule, n):
    if rule == "Gauss-Legendre":
        z, w = gauss_legendre(n)
    elif rule == "Gauss-Lobatto":
        z, w = gauss_lobatto(n)
    elif rule == "Gauss-Radau":
        z, w = gauss_radau_left(n)
    elif rule == "Fejer-I":
        z, w = fejer_type1(n)
    elif rule == "Clenshaw-Curtis":
        z, w = clenshaw_curtis(n)
    else:
        raise ValueError("Regra desconhecida")
    x, ww = map_to_interval(z, w, a, b)
    return float(np.sum(ww*f(x)))

# Referências analíticas/numericamente exatas
def I1_ref():
    F = lambda t: 0.25*np.exp(2*t)*(2*t - 1)
    return float(F(4) - F(0))
def I2_ref():
    return float(mp.si(1))  # Si(1)
def I3_ref():
    return float(mp.quad(lambda x: mp.e**(x**2), [0, 2]))
def I4_ref():
    f = lambda u: (mp.e**u - 1)/u if u != 0 else 1.0
    return float(mp.quad(f, [-1, 0]))

def main(out_dir_fig="figures", out_dir_tab="tables"):
    os.makedirs(out_dir_fig, exist_ok=True)
    os.makedirs(out_dir_tab, exist_ok=True)
    refs = {"I1": I1_ref(), "I2": I2_ref(), "I3": I3_ref(), "I4": I4_ref()}
    def f1(x): return x*np.exp(2*x)
    def f2(x):
        out = np.empty_like(x, dtype=float)
        for i, xi in enumerate(x):
            out[i] = 1.0 if abs(xi) < 1e-12 else np.sin(xi)/xi
        return out
    def f3(x): return np.exp(x**2)
    def f4(x):
        y = x - 1.0
        out = np.empty_like(x, dtype=float)
        for i, yi in enumerate(y):
            out[i] = 1.0 if abs(yi) < 1e-12 else (np.exp(yi)-1.0)/yi
        return out
    integrals = {"I1": (f1, 0.0, 4.0), "I2": (f2, -1.0, 0.0),
                 "I3": (f3, 0.0, 2.0), "I4": (f4, 0.0, 1.0)}
    rules = ["Gauss-Legendre", "Gauss-Lobatto", "Gauss-Radau", "Fejer-I", "Clenshaw-Curtis"]
    ns = [4, 8, 16, 32]
    rows = []
    for key, (f, a, b) in integrals.items():
        I_ref = refs[key]
        for rule in rules:
            for n in ns:
                try:
                    I_num = integrate_rule(f, a, b, rule, n)
                    rel = abs(I_num - I_ref)/abs(I_ref)
                except Exception:
                    I_num, rel = float("nan"), float("nan")
                rows.append({"integral": key, "method": rule, "n": n, "I_ref": I_ref, "I_num": I_num, "rel_error": rel})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir_tab, "errors_quadratures.csv"), index=False)
    # Salva tabelas LaTeX por integral
    for key in integrals.keys():
        sub = df[df["integral"]==key].pivot(index="n", columns="method", values="rel_error").sort_index()
        sub_fmt = sub.applymap(lambda v: f"{v:.2e}" if np.isfinite(v) else "--")
        latex = sub_fmt.to_latex(escape=False, caption=f"Erros relativos para {key}", label=f"tab:errors_{key}")
        with open(os.path.join(out_dir_tab, f"errors_{key}.tex"), "w") as f:
            f.write(latex)
    # Figuras (uma por integral)
    for key in integrals.keys():
        sub = df[df["integral"]==key]
        plt.figure()
        for rule in rules:
            s = sub[sub["method"]==rule].sort_values("n")
            plt.loglog(s["n"], s["rel_error"], marker="o", label=rule)
        plt.xlabel("n (pontos)"); plt.ylabel("erro relativo")
        plt.title(f"Convergência de erro – {key}")
        plt.legend(); plt.grid(True, which="both")
        plt.savefig(os.path.join(out_dir_fig, f"convergence_{key}.png"), bbox_inches="tight", dpi=180)
        plt.close()

if __name__ == "__main__":
    main()
