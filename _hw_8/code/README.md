# Exercise 01 - Chebyshev-Lobatto Quadrature

This folder contains the driver script that evaluates the four integrals from the
assignment using the helper functions implemented in the `SpectralTools` package.

## How to run

1. Open a terminal in `PTC5725/_hw_8/code`.
2. Make sure the `SpectralTools` package is available in your Julia environment
   (for local development this usually means `pkg> dev ../../SpectralTools`).
3. Install the script dependencies if needed:
   `pkg> add CSV DataFrames`.
4. Execute the script:
   ```
   julia exercise01_cheb_lobatto.jl
   ```

The script prints one table per integral (with the requested columns) and writes
`exercise01_cheb_lobatto_results.csv` in the same directory.
