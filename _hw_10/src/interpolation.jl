using SpectralTools
using Statistics

"""
    interpolate_to_uniform(x_nodes, y_nodes, F; num_points=101)

Interpola a solução espectral `F` para uma malha uniforme `num_points × num_points`
e calcula os erros em relação à solução analítica.
"""
function interpolate_to_uniform(x_nodes, y_nodes, F; num_points::Int=101)
    x_uniform = collect(range(-1.0, 1.0; length=num_points))
    y_uniform = collect(range(-1.0, 1.0; length=num_points))
    numerical = interp2D_spectral(x_nodes, y_nodes, F, x_uniform, y_uniform)
    Xuni, Yuni = grid2D(x_uniform, y_uniform)
    exact = exp.(Xuni .+ Yuni)
    error = numerical .- exact
    abs_error = abs.(error)
    max_error = maximum(abs_error)
    l2_error = sqrt(mean(abs_error .^ 2))
    return (x=x_uniform, y=y_uniform, numerical=numerical, exact=exact,
            error=error, max_error=max_error, l2_error=l2_error)
end
