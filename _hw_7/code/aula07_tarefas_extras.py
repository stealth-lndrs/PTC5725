# Python 3.11+
import numpy as np
import mpmath as mp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

mp.mp.dps = 80

def map_to_interval(x_hat, a, b):
    return 0.5 * (b - a) * x_hat + 0.5 * (a + b)

def cheb_gauss(n):
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * np.pi / (2 * n))
    w = np.full(n, np.pi / n)
    return x, w

def cheb_type2_gauss(n):
    k = np.arange(1, n + 1)
    x = np.cos(k * np.pi / (n + 1))
    w = (np.pi / (n + 1)) * (np.sin(k * np.pi / (n + 1)) ** 2)
    return x, w

def legendre_gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def integrate_on_ab(f, a, b, rule, n):
    if rule == "cheb_gauss":
        xh, wh = cheb_gauss(n)
        w_fun = lambda x: 1.0 / np.sqrt(1.0 - x * x)
    elif rule == "cheb_type2":
        xh, wh = cheb_type2_gauss(n)
        w_fun = lambda x: np.sqrt(1.0 - x * x)
    elif rule == "legendre":
        xh, wh = legendre_gauss(n)
        w_fun = lambda x: np.ones_like(x)
    else:
        raise ValueError("Unknown rule")

    x = map_to_interval(xh, a, b)
    if rule.startswith("cheb_"):
        xh_safe = np.clip(xh, -1 + 1e-15, 1 - 1e-15)
        g_vals = np.array([f(float(xx)) for xx in x]) / w_fun(xh_safe)
    else:
        g_vals = np.array([f(float(xx)) for xx in x])

    J = 0.5 * (b - a)
    return J * np.dot(wh, g_vals)

if __name__ == "__main__":
    root = Path(".")
    (root / "figures").mkdir(exist_ok=True, parents=True)
    (root / "tables").mkdir(exist_ok=True, parents=True)

    def f_exp(x): return float(np.exp(x))
    def f_sinpix(x): return float(np.sin(np.pi * x))

    tests = [("exp", f_exp, -1.0, 1.0),
             ("sin_pi_x", f_sinpix, -1.0, 1.0)]
    ns = [4, 8, 16, 32]

    records = []
    for name, f, a, b in tests:
        ref = float(mp.quad(lambda t: f(float(t)), [a, b]))
        for n in ns:
            for rule in ["cheb_gauss", "cheb_type2", "legendre"]:
                approx = float(integrate_on_ab(f, a, b, rule, n))
                rel_err = abs(approx - ref) / max(1.0, abs(ref))
                records.append({
                    "test": name, "a": a, "b": b, "n": n, "rule": rule,
                    "approx": approx, "reference": ref, "rel_error": rel_err
                })

    df = pd.DataFrame.from_records(records)
    summary = (df.groupby(["rule", "n"])["rel_error"]
                 .mean().reset_index()
                 .sort_values(["n", "rule"]))

    with open(root / "tables" / "cheb_vs_legendre_errors.tex", "w") as fh:
        fh.write(summary.rename(columns={
            "rule": "Regra",
            "n": "$n$",
            "rel_error": "Erro relativo médio"
        }).to_latex(index=False, float_format="%.3e"))

    # Figures per test
    for test_name in df["test"].unique():
        sub = df[df["test"] == test_name]
        pivoted = sub.pivot(index="n", columns="rule", values="rel_error").sort_index()
        plt.figure()
        for rule in ["cheb_gauss", "cheb_type2", "legendre"]:
            if rule in pivoted.columns:
                plt.loglog(pivoted.index.values, pivoted[rule].values, marker="o", label=rule)
        plt.xlabel("n (nós)"); plt.ylabel("Erro relativo"); plt.title(f"Convergência – {test_name}")
        plt.legend(); plt.savefig(root / "figures" / f"cheb_gauss_vs_legendre_{test_name}.png", dpi=200, bbox_inches="tight"); plt.close()

    avg = (df.groupby(["rule", "n"])["rel_error"]
             .mean().reset_index()
             .pivot(index="n", columns="rule", values="rel_error")
             .sort_index())

    plt.figure()
    for rule in ["cheb_gauss", "cheb_type2", "legendre"]:
        if rule in avg.columns:
            plt.loglog(avg.index.values, avg[rule].values, marker="o", label=rule)
    plt.xlabel("n (nós)"); plt.ylabel("Erro relativo médio (2 funções)")
    plt.title("Convergência média: Chebyshev I, Chebyshev II e Gauss–Legendre")
    plt.legend(); plt.savefig(root / "figures" / "cheb_gauss_vs_legendre.png", dpi=220, bbox_inches="tight"); plt.close()
