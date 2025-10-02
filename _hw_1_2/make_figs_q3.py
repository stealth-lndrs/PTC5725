
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diff_matrix import differentiate

out = Path(__file__).resolve().parent.parent / "figs"
out.mkdir(parents=True, exist_ok=True)

f  = lambda x: np.exp(x) * np.sin(5*x)
fp = lambda x: np.exp(x) * (np.sin(5*x) + 5*np.cos(5*x))

N = 20
x_eq = np.linspace(-1, 1, N+1)
x_ch = np.cos(np.pi * np.arange(N+1) / N)

num_eq = differentiate(x_eq, f(x_eq), k=1)
num_ch = differentiate(x_ch, f(x_ch), k=1)

err_eq = np.abs(num_eq - fp(x_eq))
err_ch = np.abs(num_ch - fp(x_ch))

plt.figure(figsize=(6.2, 4.4))
plt.plot(x_eq, fp(x_eq), label="f'(x) exata")
plt.plot(x_eq, num_eq, "--", label="f'(x) numérica (D)")
plt.title("Q3 — Derivada em nós equidistantes")
plt.xlabel("x"); plt.ylabel("f'(x)"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(out / "q3_deriv_eq.png", dpi=200); plt.close()

plt.figure(figsize=(6.2, 4.4))
plt.plot(x_ch, fp(x_ch), label="f'(x) exata")
plt.plot(x_ch, num_ch, "--", label="f'(x) numérica (D)")
plt.title("Q3 — Derivada em nós de Chebyshev-Lobatto")
plt.xlabel("x"); plt.ylabel("f'(x)"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(out / "q3_deriv_ch.png", dpi=200); plt.close()

plt.figure(figsize=(6.2, 4.4))
plt.plot(x_eq, err_eq, label="|erro|")
plt.title("Q3 — Erro |f' - (Df)| em nós equidistantes")
plt.xlabel("x"); plt.ylabel("|erro|"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(out / "q3_erro_eq.png", dpi=200); plt.close()

plt.figure(figsize=(6.2, 4.4))
plt.plot(x_ch, err_ch, label="|erro|")
plt.title("Q3 — Erro |f' - (Df)| em nós de Chebyshev-Lobatto")
plt.xlabel("x"); plt.ylabel("|erro|"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(out / "q3_erro_ch.png", dpi=200); plt.close()
