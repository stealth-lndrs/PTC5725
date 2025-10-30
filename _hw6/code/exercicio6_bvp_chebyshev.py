
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/mnt/data"
FIG_DIR = os.path.join(BASE, "figures")
TAB_DIR = os.path.join(BASE, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def cheb(n: int):
    if n == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1); c[0] = 2.0; c[-1] = 2.0
    c = c * ((-1) ** np.arange(n + 1))
    X = np.tile(x, (n + 1, 1)); dX = X - X.T
    D = (np.outer(c, 1 / c)) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

pi = np.pi
n = 31
N = n + 1

D_raw, x = cheb(n)
D1 = -D_raw             # sign correction so D1 acts like d/dx
D2 = D1 @ D1
I = np.eye(N)

f = -pi * np.sin(pi * x)
y_true = np.cos(pi * x)
L = D2 + D1 + (pi**2) * I

def impose_dirichlet(Lmat, rhs):
    A = Lmat.copy(); b = rhs.copy()
    A[-1,:] = 0.0; A[-1,-1] = 1.0; b[-1] = -1.0
    A[0,:]  = 0.0; A[0,0]  = 1.0;  b[0]  = -1.0
    return A, b

def impose_neumann(Lmat, rhs, D1):
    A = Lmat.copy(); b = rhs.copy()
    A[-1,:] = D1[-1,:]; b[-1] = 0.0
    A[0,:]  = D1[0,:];  b[0]  = 0.0
    return A, b

def impose_robin(Lmat, rhs, D1, alpha=1.0, beta=1.0, g=-1.0):
    A = Lmat.copy(); b = rhs.copy()
    A[-1,:] = alpha*np.eye(N)[-1,:] + beta*D1[-1,:]; b[-1] = g
    A[0,:]  = alpha*np.eye(N)[0,:]  + beta*D1[0,:];  b[0]  = g
    return A, b

# Solve
A_dir,b_dir = impose_dirichlet(L, f)
y_dir = np.linalg.solve(A_dir, b_dir)

A_neu,b_neu = impose_neumann(L, f, D1)
y_neu = np.linalg.solve(A_neu, b_neu)

A_rob,b_rob = impose_robin(L, f, D1, 1.0, 1.0, -1.0)
y_rob = np.linalg.solve(A_rob, b_rob)

# Errors
err_dir = float(np.max(np.abs(y_dir - y_true)))
err_neu = float(np.max(np.abs(y_neu - y_true)))
err_rob = float(np.max(np.abs(y_rob - y_true)))

# Plots
sort_idx = np.argsort(x)
xs = x[sort_idx]

def save_plot(path, y_num, title):
    plt.figure()
    plt.plot(xs, y_true[sort_idx], linestyle="--", label="Analítica: cos(pi x)")
    plt.plot(xs, y_num[sort_idx], label="Numérica")
    plt.xlabel("x"); plt.ylabel("y(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

save_plot(os.path.join(FIG_DIR, "y_dirichlet.png"), y_dir, "BVP com CC Dirichlet")
save_plot(os.path.join(FIG_DIR, "y_neumann.png"), y_neu, "BVP com CC Neumann")
save_plot(os.path.join(FIG_DIR, "y_robin.png"), y_rob, "BVP com CC Robin")

plt.figure()
plt.plot(xs, y_dir[sort_idx], label="Dirichlet")
plt.plot(xs, y_neu[sort_idx], label="Neumann")
plt.plot(xs, y_rob[sort_idx], label="Robin")
plt.xlabel("x"); plt.ylabel("y(x)")
plt.title("Comparação das soluções numéricas (Dirichlet, Neumann, Robin)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "y_comparison.png"), dpi=150)
plt.close()

# Table (CSV)
errors_df = pd.DataFrame({
    "Condicao_de_Contorno": ["Dirichlet", "Neumann", "Robin"],
    "n": [n, n, n],
    "Max_erro_infinito": [err_dir, err_neu, err_rob]
})
errors_df.to_csv(os.path.join(TAB_DIR, "ex6_max_errors.csv"), index=False)

print("Done. Figures in", FIG_DIR, "Tables in", TAB_DIR, "Errors:", err_dir, err_neu, err_rob)
