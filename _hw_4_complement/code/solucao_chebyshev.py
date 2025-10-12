import numpy as np
import matplotlib.pyplot as plt

N = 30
xi = np.cos(np.pi * np.arange(N+1) / N)
x = 2 + xi

c = np.ones(N+1); c[0] = c[-1] = 2
D = np.zeros((N+1, N+1))
for i in range(N+1):
    for j in range(N+1):
        if i != j:
            D[i,j] = (c[i]/c[j]) * (-1)**(i+j) / (xi[i]-xi[j])
    D[i,i] = -np.sum(D[i,:])

A = np.diag(x) @ D + 2*np.eye(N+1)
b = 4*x**2

# Condição de contorno correta: u(1)=2 no último nó (x=1)
A[-1,:] = 0
A[-1,-1] = 1
b[-1] = 2

u_num = np.linalg.solve(A, b)
u_ana = x**2 + 1/x**2
erro_max = np.max(np.abs(u_num - u_ana))
print(f"Erro máximo: {erro_max:.2e}")

plt.figure(figsize=(6,4))
plt.plot(x, u_ana, 'k-', label='Solução analítica')
plt.plot(x, u_num, 'ro', label='Solução numérica (Chebyshev corrigida)')
plt.xlabel('$x$'); plt.ylabel('$u(x)$')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('fig_solucao_chebyshev_corrigida.png', dpi=300)
plt.show()
