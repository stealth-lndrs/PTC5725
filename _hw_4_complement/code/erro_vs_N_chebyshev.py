import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cheb_diff_matrix(N):
    xi = np.cos(np.pi * np.arange(N+1) / N)
    c = np.ones(N+1); c[0] = c[-1] = 2
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = (c[i]/c[j]) * (-1)**(i+j) / (xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])
    return xi, D

def solve_for_N(N):
    xi, D = cheb_diff_matrix(N)
    x = 2 + xi  # map [-1,1] -> [1,3]
    A = np.diag(x) @ D + 2*np.eye(N+1)
    b = 4*x**2
    # boundary u(1)=2 at xi=-1 -> last node
    A[-1,:] = 0; A[-1,-1] = 1; b[-1] = 2
    u_num = np.linalg.solve(A, b)
    u_ana = x**2 + 1/x**2
    err_max = np.max(np.abs(u_num - u_ana))
    return err_max

# Exponential-like sequence of N (small to moderate)
Ns = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

errors = [solve_for_N(N) for N in Ns]

# Save CSV
import pandas as pd
df = pd.DataFrame({'N': Ns, 'erro_max': errors})
df.to_csv('erro_convergencia_cheb.csv', index=False)

# Plot error vs N
plt.figure(figsize=(6,4))
plt.semilogy(Ns, errors, marker='o')
plt.xlabel('N (grau do polinômio)')
plt.ylabel('Erro máximo |u_num - u_ana|')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('erro_vs_N_chebyshev.png', dpi=300)
plt.show()
