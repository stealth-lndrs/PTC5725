ENV["GKSwstype"] = "100"
using Plots

"""
    plot_surface(x, y, F, filepath; title="")

Cria um gráfico de superfície para `F(x, y)` e salva em `filepath`.
"""
function plot_surface(x, y, F, filepath; title="Solução numérica")
    plt = surface(x, y, F;
                  xlabel="x",
                  ylabel="y",
                  zlabel="F(x,y)",
                  title=title,
                  colorbar=true)
    savefig(plt, filepath)
end

"""
    plot_error_surface(x, y, E, filepath; title="")

Gera um gráfico de superfície para o erro absoluto `E` e salva a figura.
"""
function plot_error_surface(x, y, E, filepath; title="Erro absoluto")
    plt = surface(x, y, E;
                  xlabel="x",
                  ylabel="y",
                  zlabel="|erro|",
                  title=title,
                  colorbar=true)
    savefig(plt, filepath)
end

"""
    heatmap_error(x, y, E, filepath; title="")

Produz um mapa de calor bidimensional do erro absoluto.
"""
function heatmap_error(x, y, E, filepath; title="Mapa de calor do erro")
    plt = heatmap(x, y, E;
                  xlabel="x",
                  ylabel="y",
                  title=title,
                  color=:thermal,
                  colorbar_title="|erro|")
    savefig(plt, filepath)
end
