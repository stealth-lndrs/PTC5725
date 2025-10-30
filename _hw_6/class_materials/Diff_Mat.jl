using LinearAlgebra
function Generalized_Diff_Mat(xs)
    # 1. Preparação dos Dados
    xs = Float64.(xs)
    N = length(xs)
    
    # 2. Matrizes de Diferença (sem modificação)
    # dx_zeros: Matriz onde dx_zeros[i, j] = xs[i] - xs[j] (com 0 na diagonal)
    dx_zeros = xs .- xs' 
    
    # 3. Cálculo dos Coeficientes de Lagrange (aj): PRODUTO SEM A DIAGONAL
    aj = ones(N) # Inicializa aj com 1.0

    for j in 1:N
        # Para cada ponto j, o produto é (x_j - x_i) para todos i != j
        # O prod é feito apenas nos termos fora da diagonal.
        
        # A forma mais limpa é: calcular o produto de (xs[j] - xs[k]) onde k != j
        # Usamos o dX com 1s na diagonal temporariamente:
        dX_temp = copy(dx_zeros) 
        dX_temp[j, j] = 1.0 # Garante que o termo diagonal é 1 (para o produto)
        
        # O produto de cada coluna j (dims=1)
        aj[j] = prod(dX_temp, dims=1)[j]
    end
    
    # 4. Pré-cálculo da Matriz de Diferenciação D
    # Inicializamos D com zeros e calculamos APENAS os termos fora da diagonal.
    D = zeros(N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                # D[i, j] = (aj[i] / aj[j]) * (1 / (xs[i] - xs[j]))
                D[i, j] = (aj[i] / aj[j]) / dx_zeros[i, j]
            end
        end
    end

    # 5. Correção da Diagonal (D[i, i] = - sum(D[i, j]) para j != i)
    # sum(D, dims=2) calcula a soma dos termos fora da diagonal.
    # D = D - Diagonal(...) garante que D[i, i] = D[i, i] - sum(linha_i)
    
    # D_base é a matriz com D[i, i] = 0.0
    # O passo de subtração calcula D[i, i] = 0.0 - (soma dos off-diagonais)
    D = D - Diagonal(vec(sum(D, dims=2)))

    return D
end
