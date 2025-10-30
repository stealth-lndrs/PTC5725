"""
Fourier spectral derivatives and (optional) trigonometric interpolation (FFT-based)

This module provides:
- spectral_derivative_periodic: first derivative via FFT on [0, 2π)
- spectral_second_derivative_periodic: second derivative via FFT
- trig_interpolant_fft: evaluate the trigonometric interpolant via FFT coefficients
- A demo in the __main__ section that reproduces the figures and tables
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def spectral_derivative_periodic(fx):
    """
    Compute first derivative of a 2π-periodic function sampled at N equispaced x in [0, 2π).
    """
    N = fx.size
    F = np.fft.fft(fx)
    k = np.fft.fftfreq(N, d=1.0/N)  # integer wavenumbers
    dF = 1j * k * F
    return np.fft.ifft(dF).real

def spectral_second_derivative_periodic(fx):
    """
    Compute second derivative of a 2π-periodic function sampled at N equispaced x in [0, 2π).
    """
    N = fx.size
    F = np.fft.fft(fx)
    k = np.fft.fftfreq(N, d=1.0/N)
    d2F = -(k**2) * F
    return np.fft.ifft(d2F).real

def trig_interpolant_fft(x_nodes, f_nodes, x_query):
    """
    Evaluate the trigonometric interpolant (degree up to floor((N-1)/2))
    of samples f_nodes at equispaced nodes x_nodes in [0,2π) at points x_query.
    Implementation via FFT coefficients and direct evaluation.
    """
    N = len(f_nodes)
    c = np.fft.fft(f_nodes) / N
    k = np.fft.fftfreq(N, d=1.0/N)
    xq = np.atleast_1d(x_query)
    phase = np.outer(k, xq)
    s = (c[:, None] * np.exp(1j * phase)).sum(axis=0).real
    return s if np.ndim(x_query) > 0 else s.item()

# Example test function and derivatives
def f_true(x):
    return np.sin(3*x) + 0.5*np.cos(5*x)

def df_true(x):
    return 3*np.cos(3*x) - 2.5*np.sin(5*x)

def d2f_true(x):
    return -9*np.sin(3*x) - 12.5*np.cos(5*x)

if __name__ == "__main__":
    # Demo parameters
    import os
    base_dir = "."
    fig_dir = os.path.join(base_dir, "figures")
    tab_dir = os.path.join(base_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Representative derivative comparison at N=256
    N = 256
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    fx = f_true(x)
    df_fft = spectral_derivative_periodic(fx)
    d2f_fft = spectral_second_derivative_periodic(fx)
    df_ex = df_true(x)
    d2f_ex = d2f_true(x)

    plt.figure()
    plt.plot(x, df_ex, label="Analítica f'(x)")
    plt.plot(x, df_fft, '--', label="FFT f'(x)")
    plt.xlabel("x"); plt.ylabel("f'(x)"); plt.legend()
    plt.savefig(os.path.join(fig_dir, "fourier_derivative_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Convergence study
    N_list = [16, 32, 64, 128, 256, 512]
    recs = []
    for NN in N_list:
        xx = np.linspace(0, 2*np.pi, NN, endpoint=False)
        fxx = f_true(xx)
        d1 = spectral_derivative_periodic(fxx)
        d2 = spectral_second_derivative_periodic(fxx)
        d1_ex = df_true(xx)
        d2_ex = d2f_true(xx)
        l2_d1 = np.sqrt(np.mean((d1 - d1_ex)**2))
        linf_d1 = np.max(np.abs(d1 - d1_ex))
        l2_d2 = np.sqrt(np.mean((d2 - d2_ex)**2))
        linf_d2 = np.max(np.abs(d2 - d2_ex))
        recs.append(dict(N=NN, L2_err_d1=l2_d1, Linf_err_d1=linf_d1, L2_err_d2=l2_d2, Linf_err_d2=linf_d2))

    import pandas as pd
    df = pd.DataFrame(recs).sort_values("N")
    df.to_csv(os.path.join(tab_dir, "fourier_errors.csv"), index=False)

    plt.figure()
    plt.loglog(df["N"], df["Linf_err_d1"], 'o-', label="|erro|∞ f'")
    plt.loglog(df["N"], df["L2_err_d1"], 's-', label="|erro|2 f'")
    plt.loglog(df["N"], df["Linf_err_d2"], 'o--', label="|erro|∞ f''")
    plt.loglog(df["N"], df["L2_err_d2"], 's--', label="|erro|2 f''")
    plt.xlabel("N (pontos)"); plt.ylabel("Erro"); plt.legend()
    plt.savefig(os.path.join(fig_dir, "fourier_error_convergence.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Aliasing demo
    def f_high(x):
        return np.sin(12*x)
    N_small = 16
    x_small = np.linspace(0, 2*np.pi, N_small, endpoint=False)
    f_small = f_high(x_small)
    x_dense = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    # Use FFT-based interpolant from under-sampled data
    c = np.fft.fft(f_small) / N_small
    k = np.fft.fftfreq(N_small, d=1.0/N_small)
    phase = np.outer(k, x_dense)
    f_rec = (c[:, None] * np.exp(1j * phase)).sum(axis=0).real

    plt.figure()
    plt.plot(x_dense, f_high(x_dense), label="Função verdadeira sin(12x)")
    plt.plot(x_dense, f_rec, '--', label=f"Reconstrução com N={N_small} (aliased)")
    plt.scatter(x_small, f_small, s=12, label="Amostras N=16")
    plt.xlabel("x"); plt.ylabel("f(x)"); plt.legend()
    plt.savefig(os.path.join(fig_dir, "fourier_aliasing_demo.png"), dpi=160, bbox_inches="tight")
    plt.close()
