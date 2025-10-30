# Wave1Dfree.jl
using Plots
include("MinhasFunc.jl")
#using .MinhasFunc

function Wave1Dfree(n)
    
    k = 0:n
    xs = -cos.(k .* (pi/n))
    D1 = Generalized_Diff_Mat(xs)
    dt = 4/n^2
    
    # --- Configuração dos Vetores (Correta) ---
    U = @. exp(-40 * (xs - 0.4)^2)
    U[end] = 0.0
    Uold = copy(U)
    Unew = similar(U)
    
    xp = collect(range(-1, 1, length=101))
    _, Minterp = Bary_Interp(xs, xs, xp)
    
    kmax = round(Int, 3/dt)
    t = 0.0
    
    println("Iniciando a simulação...")
    
    # --- 1. REMOVEMOS O PLOT INICIAL DAQUI ---
    
    # --- Loop da Simulação ---
    for k in 0:kmax
        t += dt
        
        # 1. Calcular derivadas (estado `j`)
        Ux = D1 * U
        Ux[1] = 0.0 
        Uxx = D1 * Ux
        
        # 2. Calcular estado `j+1` em `Unew`
        @. Unew = 2*U - Uold + dt^2 * Uxx
        Unew[end] = 0.0
        
        
        # --- 3. CORREÇÃO: PLOTAR O NOVO ESTADO (`Unew`) ---
        # (Isto estava correto na última versão)
        Uplot = Minterp * Unew
            
        # --- 4. CORREÇÃO: FORÇAR UM NOVO PLOT (estilo "hold off") ---
        # Em vez de `plot!`, criamos um NOVO objeto `p` a cada iteração.
        # Isto é o seu "fechar a figura e surgir uma nova".
        p = plot(xp, Uplot, 
                 linewidth=3, 
                 ylims=(-1, 1), 
                 xlims=(-1, 1), 
                 title=string(round(t, digits=2)),
                 legend=false,
                 size=(1400, 800)
            )
        
        # Forçamos a exibição (display) deste novo plot
        display(p)
        sleep(0.01) # Ajuste a gosto

        
        # --- 5. ATUALIZAÇÃO DE ESTADO (Correta) ---
        copy!(Uold, U)
        copy!(U, Unew)
    
    end 
    
    println("Simulação concluída.")
end