import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def cheb_cgl_nodes(N):
    xi = np.cos(np.pi * np.arange(N+1) / N)
    x = 2 + xi  # map [-1,1] -> [1,3]
    return xi, x

def cheb_diff_matrix(xi):
    N = len(xi) - 1
    c = np.ones(N+1); c[0] = 2; c[-1] = 2
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i, j] = (c[i]/c[j]) * (-1)**(i+j) / (xi[i] - xi[j])
        D[i, i] = -np.sum(D[i, :])
    return D

def barycentric_weights_cgl(N):
    w = np.ones(N+1)
    w[0] = 0.5; w[-1] = 0.5
    w *= (-1)**np.arange(N+1)
    return w

def barycentric_eval_and_deriv(x_nodes, y_nodes, w, xq):
    diff = xq - x_nodes
    # exact node
    hit = np.where(np.isclose(diff, 0.0, atol=0, rtol=0))[0]
    if hit.size > 0:
        j = hit[0]
        return float(y_nodes[j]), float(0.0)
    r  = w / diff
    s1 = np.sum(r)
    s2 = np.dot(r, y_nodes)
    p  = s2 / s1
    rp = -w / (diff**2)
    t1 = np.sum(rp)
    t2 = np.dot(rp, y_nodes)
    dp = (t2*s1 - s2*t1) / (s1*s1)
    return float(p), float(dp)

def solve_ode_collocation(N):
    xi, x = cheb_cgl_nodes(N)
    D = cheb_diff_matrix(xi)  # linear map: dx/dxi = 1
    A = np.diag(x) @ D + 2*np.eye(N+1)
    b = 4 * x**2
    # boundary u(1)=2 at xi=-1 (last node)
    A[-1,:] = 0.0; A[-1,-1] = 1.0; b[-1] = 2.0
    u_num = np.linalg.solve(A, b)
    return x, u_num

def newton_barycentric_root(x_nodes, y_nodes, w, x0, tol=1e-13, maxit=50):
    xk = float(x0)
    hist = []
    for k in range(maxit):
        pk, dpk = barycentric_eval_and_deriv(x_nodes, y_nodes, w, xk)
        fk = pk - 4.0
        hist.append((k, xk, fk, dpk))
        if abs(fk) < tol:
            return xk, True, hist
        if dpk == 0.0 or not np.isfinite(dpk):
            return xk, False, hist
        xk1 = xk - fk/dpk
        # keep iterate reasonable in [1,3]
        if (xk1 < 1.0) or (xk1 > 3.0) or (not np.isfinite(xk1)):
            xk1 = 0.5*(xk + np.clip(xk1, 1.0, 3.0))
        xk = xk1
    return xk, False, hist

if __name__ == "__main__":
    N = 30
    x_nodes, u_num = solve_ode_collocation(N)
    w = barycentric_weights_cgl(N)
    x0 = 2.0
    x_root_num, ok, hist = newton_barycentric_root(x_nodes, u_num, w, x0, tol=1e-13, maxit=50)

    x_root_ana = math.sqrt(2.0 + math.sqrt(3.0))
    abs_err = abs(x_root_num - x_root_ana)

    # Save history
    df_hist = pd.DataFrame(hist, columns=["iter", "x_k", "f(x_k)=pN-4", "pN'(x_k)"])
    df_hist.to_csv("tables/newton_barycentric_history.csv", index=False)

    # Plot
    xx = np.linspace(1.5, 2.3, 400)
    yy_bary = np.zeros_like(xx)
    for i, xv in enumerate(xx):
        pv, _ = barycentric_eval_and_deriv(x_nodes, u_num, w, xv)
        yy_bary[i] = pv
    yy_ana = xx**2 + 1/xx**2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(xx, yy_ana, label='Analítica $x^2 + x^{-2}$')
    plt.plot(xx, yy_bary, '--', label='Interpolante baricêntrico $p_N(x)$')
    plt.axhline(4.0, linestyle=':', label='$u=4$')
    plt.axvline(x_root_ana, linestyle=':', label='$x_k$ analítico')
    plt.axvline(x_root_num, linestyle='--', label='$x_k$ numérico')
    plt.xlabel('$x$'); plt.ylabel('$u(x)$')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("figures/fig_root_barycentric_newton.png", dpi=300)

    with open("result_barycentric_newton.txt", "w") as f:
        f.write(f"N = {N}\n")
        f.write(f"Root (analytic): {x_root_ana:.16f}\n")
        f.write(f"Root (barycentric+Newton): {x_root_num:.16f}\n")
        f.write(f"Absolute error: {abs_err:.3e}\n")
        f.write(f"Converged: {ok}\n")
        f.write("Stopping criterion: |p_N(x_k)-4| < 1e-13 or max 50 iterations\n")
