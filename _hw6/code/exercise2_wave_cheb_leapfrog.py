# exercise2_wave_cheb_leapfrog.py
import numpy as np

def cheb(N):
    if N == 0:
        x = np.array([1.0])
        D = np.array([[0.0]])
        return x, D
    k = np.arange(0, N+1)
    x = np.cos(np.pi * k / N)
    c = np.ones(N+1)
    c[0] = 2; c[-1] = 2
    c = c * ((-1)**k)
    X = np.tile(x, (N+1,1))
    dX = X - X.T
    D = (np.outer(c, 1/c)) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    return x, D

def u0(x):
    return np.exp(-40.0*(x-0.4)**2)

def v0(x):
    return np.zeros_like(x)

def leapfrog_wave(D, D2, x, u0_vec, v0_vec, dt, nt, bc_type="dirichlet"):
    Np1 = len(x)
    U = np.zeros((nt+1, Np1))
    V = np.zeros((nt+1, Np1))
    U[0] = u0_vec.copy()
    V[0] = v0_vec.copy()

    u = U[0].copy()
    v = V[0].copy()

    if bc_type == "dirichlet":
        u[0] = 0.0; u[-1] = 0.0
        v[0] = 0.0; v[-1] = 0.0
    elif bc_type == "neumann":
        u[0] = u[1]; u[-1] = u[-2]
        v[0] = v[1]; v[-1] = v[-2]

    a = D2 @ u
    u_prev = u.copy()
    u = u + dt*v + 0.5*(dt**2)*a

    if bc_type == "dirichlet":
        u[0] = 0.0; u[-1] = 0.0
    else:
        u[0] = u[1]; u[-1] = u[-2]

    U[1] = u.copy()
    V[1] = (U[1]-U[0])/dt

    for n in range(1, nt):
        a = D2 @ u
        u_next = 2*u - u_prev + (dt**2)*a

        if bc_type == "dirichlet":
            u_next[0] = 0.0; u_next[-1] = 0.0
        else:
            u_next[0] = u_next[1]; u_next[-1] = u_next[-2]

        U[n+1] = u_next.copy()
        V[n+1] = (U[n+1]-U[n-1])/(2*dt)

        u_prev, u = u, u_next

    return U, V
