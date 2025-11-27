#!/usr/bin/env julia

using SpectralTools
using CSV
using DataFrames
using Printf

const QUAD_ORDERS = [10, 20, 40, 80, 160]
const EXERCISE_FUNCTIONS = [
    ("f₁(x) = 1", x -> 1.0, 2.0),
    ("f₂(x) = x^2", x -> x^2, 2 / 3),
    ("f₃(x) = (2x^2 - 1)^2", x -> (2x^2 - 1)^2, 14 / 15),
    ("f₄(x) = sqrt(1 - x^2)", x -> sqrt(1 - x^2), pi / 2),
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

    for (label, f, exact) in EXERCISE_FUNCTIONS
        println()
        println(label)
        @printf("%5s  %16s  %16s  %14s  %14s\n", "n", "numeric_value", "exact_value", "abs_error", "rel_error")
        for n in QUAD_ORDERS
            approx = cheb_lobatto_quadrature(f, n)
            abs_err = abs(approx - exact)
            rel_err = abs_err / abs(exact)
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
    csv_path = joinpath(@__DIR__, "results_exercise01_cheb_lobatto.csv")
    CSV.write(csv_path, results)
    println()
    println("Results saved to $(csv_path)")
end

main()
