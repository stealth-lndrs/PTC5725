
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from lagrange_interpolation import lagrange_interpolation

out = Path(__file__).resolve().parent.parent / "figs"
out.mkdir(parents=True, exist_ok=True)

tests = {
    "exp(-x)":       lambda x: np.exp(-x),
    "sinpi_x":       lambda x: np.sin(np.pi*x),
    "cospi_x":       lambda x: np.cos(np.pi*x),
    "poly_6":        lambda x: 32*x**6 - 48*x**4 + 18*x**2 - 1,
}

x_nodes = np.linspace(-1.0, 1.0, 12)
x_eval  = -1.0 + 2.0*np.arange(1, 121)/120.0

def metrics(y_hat, y_true, x):
    err = y_hat - y_true
    abs_err = np.abs(err)
    linf = abs_err.max()
    rms  = np.sqrt(np.mean(err**2))
    mse  = np.mean(err**2)
    l1   = np.trapz(abs_err, x)
    return linf, rms, l1, mse

rows = []
for name, f in tests.items():
    y_nodes = f(x_nodes)
    y_true  = f(x_eval)
    y_hat   = lagrange_interpolation(x_nodes, y_nodes, x_eval)
    linf, rms, l1, mse = metrics(y_hat, y_true, x_eval)
    rows.append([name, linf, rms, l1, mse])

    xx = np.linspace(-1, 1, 800)
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(xx, f(xx), label="Função")
    yy = lagrange_interpolation(x_nodes, y_nodes, xx)
    plt.plot(xx, yy, "--", label="Interpolação (Lagrange)")
    plt.plot(x_nodes, y_nodes, "o", label="Nós (equispaciados)")
    plt.title(f"Q1 — Interpolação Lagrange — {name}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out / f"q1_{name}.png", dpi=200)
    plt.close()

with open(out / "q1_metrics.csv", "w", newline="", encoding="utf-8") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["func", "Linf", "RMS", "L1", "MSE"])
    w.writerows(rows)
