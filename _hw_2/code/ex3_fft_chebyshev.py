
import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_lobatto_nodes, dct_type1_via_fft, cheb_reconstruct_from_coeffs

def main():
    f = lambda x: np.exp(-x)*np.sin(np.pi*x)
    for n in [32, 64, 128, 256]:
        x = cheb_lobatto_nodes(n); y = f(x)
        c = dct_type1_via_fft(y) * (2.0/n)
        c[0] *= 0.5; c[-1] *= 0.5
        plt.figure(); markerline, stemlines, baseline = plt.stem(np.arange(len(c)), np.abs(c))
        plt.title(f"Exercicio 3: |coef Chebyshev| via FFT (n={n})")
        plt.xlabel("k"); plt.ylabel("|c_k|")
        plt.savefig(f"../figures/ex3_coeffs_n{n}.png", dpi=150, bbox_inches="tight")
        y_rec = cheb_reconstruct_from_coeffs(x, c)
        err = np.linalg.norm(y - y_rec, 2)/np.sqrt(len(y))
        print(f"n={n} L2/sqrt(N) error = {err:.3e}")

if __name__ == "__main__":
    main()
