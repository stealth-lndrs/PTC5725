
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from lagrange_interpolation import lagrange_interpolation

def run_experiment(N=10, a=-1.0, b=1.0, n_eval=800, outdir="figs"):
    """
    Compara nós equidistantes vs Chebyshev-Lobatto para f(x)=1/(1+16x^2) em [a,b],
    gera figuras e calcula métricas de erro (L-inf, RMS, L1 e MSE).

    Salva:
      - figs/fig_equidistante.png
      - figs/fig_chebyshev.png
      - figs/metrics.csv
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    f = lambda x: 1.0/(1.0 + 16.0*x**2)  # função de Runge
    x_eval = np.linspace(a, b, n_eval)
    true_vals = f(x_eval)

    # Caso A: nós equidistantes
    x_eq = np.linspace(a, b, N+1)
    y_eq = f(x_eq)
    y_eq_eval = lagrange_interpolation(x_eq, y_eq, x_eval)

    # Caso B: nós de Chebyshev-Lobatto
    k = np.arange(N+1)
    x_ch = np.cos(np.pi * k / N)  # já em [-1,1]
    y_ch = f(x_ch)
    y_ch_eval = lagrange_interpolation(x_ch, y_ch, x_eval)

    # Métricas
    def metrics(y_hat, y_true, x):
        err = y_hat - y_true
        abs_err = np.abs(err)
        linf = abs_err.max()
        rms = np.sqrt(np.mean(err**2))
        mse = np.mean(err**2)
        l1 = np.trapz(abs_err, x)  # integral do erro absoluto
        return linf, rms, l1, mse

    linf_eq, rms_eq, l1_eq, mse_eq = metrics(y_eq_eval, true_vals, x_eval)
    linf_ch, rms_ch, l1_ch, mse_ch = metrics(y_ch_eval, true_vals, x_eval)

    # Figuras
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(x_eval, true_vals, label="Função original")
    plt.plot(x_eval, y_eq_eval, linestyle="--", label="Interpolação (equidistante)")
    plt.plot(x_eq, y_eq, "o", label="Nós equidistantes")
    plt.title("Nós equidistantes (fenômeno de Runge)")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out/"fig_equidistante.png", dpi=200); plt.close()

    plt.figure(figsize=(6.2, 4.4))
    plt.plot(x_eval, true_vals, label="Função original")
    plt.plot(x_eval, y_ch_eval, linestyle="--", label="Interpolação (Chebyshev)")
    plt.plot(x_ch, y_ch, "o", label="Nós de Chebyshev")
    plt.title("Nós de Chebyshev (erro reduzido)")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out/"fig_chebyshev.png", dpi=200); plt.close()

    # CSV com métricas
    with open(out/"metrics.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["grid", "Linf", "RMS", "L1", "MSE"])  # cabeçalho
        w.writerow(["equidistante", linf_eq, rms_eq, l1_eq, mse_eq])
        w.writerow(["chebyshev",   linf_ch, rms_ch, l1_ch, mse_ch])

    # Resumo
    summary = {
        "N": N,
        "n_eval": n_eval,
        "equidistante": {"Linf": linf_eq, "RMS": rms_eq, "L1": l1_eq, "MSE": mse_eq},
        "chebyshev":   {"Linf": linf_ch, "RMS": rms_ch, "L1": l1_ch, "MSE": mse_ch},
        "figs": [str(out/"fig_equidistante.png"), str(out/"fig_chebyshev.png")],
        "csv": str(out/"metrics.csv"),
    }
    return summary

if __name__ == "__main__":
    info = run_experiment(N=10, a=-1.0, b=1.0, n_eval=20001, outdir="figs")
    import json
    print(json.dumps(info, indent=2, ensure_ascii=False))
