
import numpy as np
import matplotlib.pyplot as plt
from utils import cheb_basis_matrix, clenshaw_curtis_weights

def main():
    n = 64
    x, B = cheb_basis_matrix(n)
    w = clenshaw_curtis_weights(n)
    W = np.diag(w)
    G = B.T @ W @ B
    d = np.diag(G).copy()
    d[d<=0] = np.min(d[d>0]) if np.any(d>0) else 1.0
    Gn = (B/np.sqrt(d)).T @ W @ (B/np.sqrt(d))
    plt.figure(); plt.imshow(np.abs(G), aspect='auto', origin='lower'); plt.colorbar()
    plt.title("Extensao: Matriz de Gram (Chebyshev, CC)")
    plt.xlabel("m"); plt.ylabel("n")
    plt.savefig("../figures/ext_ortho_gram.png", dpi=150, bbox_inches="tight")
    plt.figure(); plt.imshow(np.abs(Gn), aspect='auto', origin='lower', vmin=0, vmax=1.5); plt.colorbar()
    plt.title("Extensao: Gram Normalizada (~ identidade)")
    plt.xlabel("m"); plt.ylabel("n")
    plt.savefig("../figures/ext_ortho_gram_norm.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
