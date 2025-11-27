# Exercise 01 — Chebyshev–Lobatto Quadrature

This exercise evaluates the four functions

```
f₁(x) = 1
f₂(x) = x²
f₃(x) = (2x² - 1)²
f₄(x) = √(1 - x²)
```

over the interval ``[-1, 1]`` using the Chebyshev–Lobatto (Clenshaw–Curtis)
quadrature implemented in the `SpectralTools` package. The rule uses the nodes
``x_k = cos(kπ/n)`` and weights
``w₀ = w_n = π/(2n)`` and ``w_k = π/n`` for ``k = 1,…,n-1``.

## Running the script

1. Open a terminal in `PTC5725/_hw_8/code`.
2. Make sure the local `SpectralTools` package is available
   (e.g. `pkg> dev ../../SpectralTools`).
3. Install the exercise dependencies if needed:
   `pkg> add CSV DataFrames`.
4. Run:
   ```
   julia --project=. exercise01_cheb_lobatto.jl
   ```

The script prints tables with `n`, numerical approximation, analytical value,
and the absolute/relative errors for each function. It also saves the combined
results to `results_exercise01_cheb_lobatto.csv` in the same directory so you
can inspect or plot the data later.
