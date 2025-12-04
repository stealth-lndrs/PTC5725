import Pkg
Pkg.activate(dirname(@__DIR__))

using SpectralTools
using Statistics
using DelimitedFiles
using Printf

include("solve_anisotropic.jl")
include("solve_isotropic.jl")
include("interpolation.jl")
include("plots.jl")

"""
    main()

Executa todos os casos solicitados, gera interpolações, salva dados e figuras.
"""
function main()
    project_root = dirname(@__DIR__)
    ensure_output_dirs(project_root)
    cases = Dict(
        "anisotropic" => solve_anisotropic_case(),
        "isotropic_15" => solve_isotropic_15(),
        "isotropic_20" => solve_isotropic_20()
    )

    metrics = Dict{String, NamedTuple}()
    for (label, data) in cases
        metrics[label] = process_case(label, data, project_root)
    end
    write_summary(metrics, project_root)
end

"""
    ensure_output_dirs(root)

Garante a existência das pastas `data/` e `figures/` em `root`.
"""
function ensure_output_dirs(root::AbstractString)
    mkpath(joinpath(root, "data"))
    mkpath(joinpath(root, "figures"))
end

"""
    process_case(label, data, root)

Calcula erros, salva dados, gera figuras e interpolações.
Retorna as métricas consolidadas para o caso `label`.
"""
function process_case(label::AbstractString, data, root::AbstractString)
    exact = exp.(data.X .+ data.Y)
    error = data.solution .- exact
    abs_error = abs.(error)
    max_error = maximum(abs_error)
    l2_error = sqrt(mean(abs_error .^ 2))
    save_case_data(label, data, exact, error, root)
    interp = interpolate_to_uniform(data.x, data.y, data.solution)
    save_interpolation_data(label, interp, root)
    generate_figures(label, data, abs_error, interp, root)
    return (max_error=max_error, l2_error=l2_error,
            interp_max=interp.max_error, interp_l2=interp.l2_error)
end

"""
    save_case_data(label, data, exact, error, root)

Salva solução, referência analítica e erro na pasta `data/`.
"""
function save_case_data(label::AbstractString, data, exact, error, root)
    data_dir = joinpath(root, "data")
    save_matrix(joinpath(data_dir, "solution_$(label).csv"), data.solution)
    save_matrix(joinpath(data_dir, "exact_$(label).csv"), exact)
    save_matrix(joinpath(data_dir, "error_$(label).csv"), error)
end

"""
    save_interpolation_data(label, interp, root)

Persist e resultados da malha uniforme (solução e erro).
"""
function save_interpolation_data(label::AbstractString, interp, root)
    data_dir = joinpath(root, "data")
    save_matrix(joinpath(data_dir, "interp_solution_$(label).csv"), interp.numerical)
    save_matrix(joinpath(data_dir, "interp_error_$(label).csv"), interp.error)
end

"""
    save_matrix(path, A)

Exporta a matriz `A` para CSV via `DelimitedFiles.writedlm`.
"""
function save_matrix(path::AbstractString, A)
    writedlm(path, A, ',')
end

"""
    generate_figures(label, data, abs_error, interp, root)

Cria e salva as figuras da solução, erro na malha de Chebyshev e mapa de calor
na malha uniforme.
"""
function generate_figures(label::AbstractString, data, abs_error, interp, root)
    fig_dir = joinpath(root, "figures")
    plot_surface(data.x, data.y, data.solution,
                 joinpath(fig_dir, "solution_$(label).png");
                 title="Solução $(label)")
    plot_error_surface(data.x, data.y, abs_error,
                       joinpath(fig_dir, "error_surface_$(label).png");
                       title="Erro absoluto $(label)")
    heatmap_error(interp.x, interp.y, abs.(interp.error),
                  joinpath(fig_dir, "error_heatmap_$(label).png");
                  title="Erro interpolado $(label)")
end

"""
    write_summary(metrics, root)

Gera um arquivo `data/summary.txt` com as métricas principais.
"""
function write_summary(metrics::Dict{String, NamedTuple}, root)
    summary_path = joinpath(root, "data", "summary.txt")
    open(summary_path, "w") do io
        for key in sort(collect(keys(metrics)))
            m = metrics[key]
            println(io, "Caso: $key")
            @printf(io, "  Erro máximo (malha Chebyshev): %.4e\n", m.max_error)
            @printf(io, "  Erro L2 (malha Chebyshev): %.4e\n", m.l2_error)
            @printf(io, "  Erro máximo (malha uniforme): %.4e\n", m.interp_max)
            @printf(io, "  Erro L2 (malha uniforme): %.4e\n\n", m.interp_l2)
        end
    end
end

main()
