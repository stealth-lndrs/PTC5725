#!/usr/bin/env julia

using SpectralTools
using CSV
using DataFrames
using Printf

const QUAD_ORDERS = [10, 20, 40, 80, 160]
const INTEGRALS = [
    ("Integral of 1", x -> 1.0, 2.0),
    ("Integral of x", x -> x, 0.0),
    ("Integral of x^2", x -> x^2, 2 / 3),
    ("Integral of sqrt(1 - x^2)", x -> sqrt(1 - x^2), pi / 2),
]

function evaluate_integrals()
    results = DataFrame(
        integrand = String[],
        n = Int[],
        numeric_value = Float64[],
        exact_value = Float64[],
        abs_error = Float64[],
        rel_error = Float64[],
    )

    for (label, f, exact) in INTEGRALS
        println()
        println(label)
        @printf("%5s  %16s  %16s  %14s  %14s\n", "n", "numeric_value", "exact_value", "abs_error", "rel_error")
        for n in QUAD_ORDERS
            approx = cheb_lobatto_quadrature(f, n)
            abs_err = abs(approx - exact)
            rel_err = exact == 0.0 ? NaN : abs_err / abs(exact)
            @printf("%5d  %16.8f  %16.8f  %14.6e  %14.6e\n", n, approx, exact, abs_err, rel_err)
            push!(results, (
                integrand = label,
                n = n,
                numeric_value = approx,
                exact_value = exact,
                abs_error = abs_err,
                rel_error = rel_err,
            ))
        end
    end

    return results
end

function main()
    results = evaluate_integrals()
    csv_path = joinpath(@__DIR__, "exercise01_cheb_lobatto_results.csv")
    CSV.write(csv_path, results)
    println()
    println("Results saved to $(csv_path)")
end

main()
