
import numpy as np, time, json, matplotlib.pyplot as plt
from utils import cheb_lobatto_nodes, dct_type1_via_fft

def coeffs_from_func(f, n):
    x = cheb_lobatto_nodes(n); y = f(x)
    c = dct_type1_via_fft(y) * (2.0/n); c[0] *= 0.5; c[-1] *= 0.5
    return c

def main():
    sizes = [64,128,256,512,1024,2048,4096]
    f = lambda x: np.exp(-x)*np.sin(np.pi*x)
    times = []
    for n in sizes:
        reps = 8 if n<=1024 else 5
        t0 = time.time()
        for _ in range(reps):
            _ = coeffs_from_func(f, n)
        t1 = time.time()
        times.append((n, (t1-t0)/reps))
    with open("ext_perf_python.json","w") as fj:
        json.dump({"times": times}, fj, indent=2)
    plt.figure(); plt.loglog([n for n,_ in times],[t for _,t in times], marker='o')
    plt.xlabel("n (log)"); plt.ylabel("tempo (s) (log)")
    plt.title("Extensao: Performance Python (FFT Chebyshev)")
    plt.savefig("../figures/ex3_timing.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
