
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def cheb(N):
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    k = np.arange(0, N + 1)
    x = np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T + np.eye(N + 1)
    D = np.outer(c, 1.0 / c) / dX
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def solve_wave_mixed(N=25, t_end=3.0):
    D, x = cheb(N)
    D2 = D @ D
    dt = 4.0 / (N**2)
    num_steps = int(np.ceil(t_end / dt))
    dt = t_end / num_steps

    u0 = np.exp(-40.0 * (x - 0.4) ** 2)
    v0 = np.zeros_like(u0)

    DN = D[-1, :].copy()
    def enforce_neumann(u):
        coeff_N = DN[-1]
        rhs = -np.dot(DN[:-1], u[:-1])
        if abs(coeff_N) < 1e-12:
            u[-1] = u[-2]
        else:
            u[-1] = rhs / coeff_N
        return u

    a0 = D2 @ u0
    a0[0] = 0.0
    a0[-1] = 0.0

    u_nm1 = u0.copy()
    u_n = u0 + dt * v0 + 0.5 * (dt**2) * a0
    u_nm1[0] = 0.0
    u_n[0] = 0.0
    u_nm1 = enforce_neumann(u_nm1)
    u_n = enforce_neumann(u_n)

    store_every = max(1, int(np.ceil((t_end / dt) / 200)))
    times = [0.0]
    U_hist = [u_nm1.copy()]

    for step in range(1, num_steps + 1):
        a = D2 @ u_n
        a[0] = 0.0
        a[-1] = 0.0
        u_np1 = 2.0 * u_n - u_nm1 + (dt**2) * a
        u_np1[0] = 0.0
        u_np1 = enforce_neumann(u_np1)
        u_nm1, u_n = u_n, u_np1
        if step % store_every == 0 or step == num_steps:
            times.append(step * dt)
            U_hist.append(u_n.copy())

    return x, np.array(times), np.array(U_hist)

if __name__ == "__main__":
    x, T, U_hist = solve_wave_mixed(N=25, t_end=3.0)
    # Timeline surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    TT, XX = np.meshgrid(T, x, indexing='ij')
    ax.plot_surface(XX, TT, U_hist, linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("U(x,t)")
    ax.set_title("Wave equation with mixed BCs (fixed at x=-1, free at x=+1)")
    fig.tight_layout()
    plt.savefig("wave_fixed_free_timeline.png", dpi=200)
    plt.close(fig)
    # Snapshots
    fig2 = plt.figure(figsize=(8, 4.8))
    plt.plot(x, U_hist[0, :], label="t=0")
    plt.plot(x, U_hist[-1, :], label=f"t={T[-1]:.2f}")
    plt.xlabel("x"); plt.ylabel("U(x,t)")
    plt.title("Initial vs final displacement (fixed-free string)")
    plt.legend()
    fig2.tight_layout()
    plt.savefig("wave_fixed_free_snapshots.png", dpi=200)
    plt.close(fig2)
