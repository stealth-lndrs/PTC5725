ENV["GKSwstype"] = "100"  # headless plotting

using SpectralTools
using Printf
using Interpolations
using Plots

fig_dir = normpath(joinpath(@__DIR__, "..", "figures"))
isdir(fig_dir) || mkpath(fig_dir)

f_fun(x, y) = 10 * sin(8 * x * (y - 1))
Ns = [12, 16, 24, 32]
fine_axis = collect(range(-1.0, 1.0, length = 200))

results = Vector{NamedTuple}(undef, length(Ns))

for (idx, N) in enumerate(Ns)
    U = nothing
    x = nothing
    y = nothing
    solve_time = @elapsed begin
        U, x, y = poisson_chebyshev_2d(f_fun, N)
    end

    sorted_axis = nothing
    sorted_values = nothing
    interpolant = nothing
    fine_vals = nothing
    interp_time = @elapsed begin
        sorted_axis = reverse(x)
        sorted_values = reverse(reverse(U, dims = 1), dims = 2)
        interpolant = interpolate((sorted_axis, sorted_axis), sorted_values, Gridded(Linear()))
        fine_vals = [interpolant(yv, xv) for yv in fine_axis, xv in fine_axis]
    end

    println(@sprintf("N = %d -> solve %.4f s | interpolate %.4f s", N, solve_time, interp_time))

    surf_path = joinpath(fig_dir, @sprintf("surface_N%02d.png", N))
    heat_path = joinpath(fig_dir, @sprintf("heatmap_N%02d.png", N))

    @time begin
        surface_plot = surface(fine_axis, fine_axis, fine_vals;
            xlabel = "x", ylabel = "y", zlabel = "F", title = @sprintf("Poisson solution (N=%d)", N))
        savefig(surface_plot, surf_path)
        heat_plot = heatmap(fine_axis, fine_axis, fine_vals;
            xlabel = "x", ylabel = "y", title = @sprintf("Heatmap (N=%d)", N), color = :viridis)
        savefig(heat_plot, heat_path)
    end

    results[idx] = (N = N, axis = sorted_axis, fine_values = fine_vals)
end

println("\nComparing successive discretizations on the fine grid:")
for i in 1:(length(results) - 1)
    res_a = results[i]
    res_b = results[i + 1]
    diff_err = maximum(abs.(res_a.fine_values .- res_b.fine_values))
    println(@sprintf("N = %d vs N = %d -> max |Δu| = %.4e", res_a.N, res_b.N, diff_err))
end
