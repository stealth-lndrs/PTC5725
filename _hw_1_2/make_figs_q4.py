
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from barycentric import bary_weights_cheb2, bary_eval

out = Path(__file__).resolve().parent.parent / "figs"
out.mkdir(parents=True, exist_ok=True)

f = lambda x: 1.0/(1.0 + 16.0*x**2)
N = 10
x = np.cos(np.pi * np.arange(N+1) / N)
y = f(x)
w = bary_weights_cheb2(N)

xq = np.linspace(-1, 1, 800)
yq = bary_eval(x, y, xq, w=w)

plt.figure(figsize=(6.2, 4.4))
plt.plot(xq, f(xq), label="Função")
plt.plot(xq, yq, "--", label="Interpolação (baricêntrica)")
plt.plot(x, y, "o", label="Nós Chebyshev-Lobatto")
plt.title("Q4 — Interpolação baricêntrica (Chebyshev-Lobatto)")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(out / "q4_bary_cheb.png", dpi=200); plt.close()
