
import numpy as np

def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    """
    Avalia o polinômio interpolador de Lagrange nos pontos x_eval.

    Parâmetros
    ----------
    x_nodes : array-like
        Nós conhecidos x_j (distintos).
    y_nodes : array-like
        Valores f(x_j) correspondentes.
    x_eval : array-like
        Pontos nos quais avaliar o interpolador.

    Retorno
    -------
    y_eval : np.ndarray
        Valores interpolados em x_eval.
    """
    x_nodes = np.array(x_nodes, dtype=float)
    y_nodes = np.array(y_nodes, dtype=float)
    x_eval  = np.array(x_eval,  dtype=float)
    n = len(x_nodes)

    y_eval = np.zeros_like(x_eval, dtype=float)

    # Soma f(x_j) * l_j(x), onde l_j(x) = prod_{m != j} (x - x_m) / (x_j - x_m)
    for j in range(n):
        lj = np.ones_like(x_eval, dtype=float)
        for m in range(n):
            if m != j:
                lj *= (x_eval - x_nodes[m]) / (x_nodes[j] - x_nodes[m])
        y_eval += y_nodes[j] * lj

    return y_eval


if __name__ == "__main__":
    # Exemplo mínimo de uso
    f = lambda x: 1.0/(1.0 + 16.0*x**2)  # função de Runge
    xj = np.linspace(-1.0, 1.0, 5)       # nós
    yj = f(xj)                           # valores f(x_j)
    xk = np.linspace(-1.0, 1.0, 100)     # pontos de avaliação
    yk = lagrange_interpolation(xj, yj, xk)
    # Mostra um resumo
    print("Interpolação concluída:", dict(nos=len(xj), avaliacoes=len(xk)))
