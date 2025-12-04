using SpectralTools

"""
    solve_rectangular_case(nx, ny)

Resolve o problema ``(\nabla^2 + I)F = 3e^{x+y}`` em uma malha
de Chebyshev com `nx` pontos em ``x`` e `ny` pontos em ``y``.
Retorna um `NamedTuple` contendo a solução numérica e as malhas usadas.
"""
function solve_rectangular_case(nx::Int, ny::Int)
    x = cheb_lobatto_nodes(nx)
    y = cheb_lobatto_nodes(ny)
    _, D2x = cheb_D_matrices(nx)
    _, D2y = cheb_D_matrices(ny)
    X, Y = grid2D(x, y)
    rhs = 3 .* exp.(X .+ Y)
    boundary_values = exp.(X .+ Y)
    mask = build_internal_mask(size(X))
    A = laplacian_plus_I_operator(D2x, D2y)
    Ared, bred, _ = apply_dirichlet!(A, vec(rhs), mask, boundary_values)
    F_internal = solve_poisson_like(Ared, bred)
    solution = rebuild_solution(F_internal, boundary_values, mask)
    return (solution=solution, x=x, y=y, X=X, Y=Y)
end

"""
    solve_anisotropic_case()

Resolve o caso anisotrópico especificado (15 pontos em ``x`` e 20 em ``y``).
"""
solve_anisotropic_case() = solve_rectangular_case(15, 20)

"""
    build_internal_mask(size_tuple)

Cria a máscara booleana dos pontos internos (excluindo o contorno)
para uma malha de dimensões `size_tuple`.
"""
function build_internal_mask(size_tuple::Tuple{Int, Int})
    ny, nx = size_tuple
    mask = trues(ny, nx)
    mask[1, :] .= false
    mask[end, :] .= false
    mask[:, 1] .= false
    mask[:, end] .= false
    return mask
end
