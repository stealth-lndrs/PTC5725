# Spectral Methods — Complementary Task (Aula 4)

**Resumo rápido do que foi feito**

- **Problema:** Resolver analiticamente e numericamente a EDO
  \(x\,u'(x)+2u=4x^2\) em \([1,3]\) com \(u(1)=2\).
- **Solução analítica:** \(u(x)=x^2+x^{-2}\). Ponto onde \(u(x)=4\): \(x_k=\sqrt{2+\sqrt{3}}\).
- **Solução numérica:** Colocação **Chebyshev–Gauss–Lobatto (CGL)**: montagem de \(D\), sistema \(\mathrm{diag}(x)D+2I\), imposição de \(u(1)=2\), comparação com a solução analítica.
- **Estudo de convergência:** Erro \(\|u_{num}-u_{ana}\|_\infty\) vs \(N\) mostra convergência espectral e **saturação de erro** (piso de máquina).
- **Item (b):** Raiz via **interpolante baricêntrico + Newton–Raphson** para \(p_N(x)-4=0\) — coincidência com o valor analítico até \(\approx 10^{-16}\).
- **Item (c):** Equivalências para \(x_k=\sqrt{2+\sqrt{3}}\) e versão trigonométrica \(x_k=2\cos(\pi/12)\).

---

## Estrutura de pastas

```
code/
  barycentric_newton_root.py        # raiz de u(x)=4 via p_N + Newton
  erro_vs_N_chebyshev.py            # estudo de convergência do erro vs N
  solucao_chebyshev_corrigida.py    # colocação CGL e figura comparativa

figures/
  fig_solucao_chebyshev_corrigida.png
  erro_vs_N_chebyshev.png
  fig_root_barycentric_newton.png
  fig_erro_teorico_componentes_final.png

tables/
  erro_convergencia_cheb.csv
  newton_barycentric_history.csv
```

---

## Como rodar (ambiente Python)

1. **Criar venv e ativar**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar dependências**
   Crie `requirements.txt` com:
   ```txt
   numpy>=1.24
   matplotlib>=3.7
   pandas>=2.0
   ```
   E instale:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Executar scripts**
   ```bash
   # solução numérica e figura comparativa
   python code/solucao_chebyshev_corrigida.py

   # estudo de convergência (gera CSV e figura)
   python code/erro_vs_N_chebyshev.py

   # raiz via interpolante baricêntrico + Newton (gera CSV e figura)
   python code/barycentric_newton_root.py
   ```

> **Dica LaTeX:** para exibir código com `minted`, compile com `--shell-escape` e, se o código for longo, **não** use float `[H]`; prefira `\captionof{listing}+\inputminted` (sem float) para permitir quebra de página.

---

## Resultados principais (item b)

- Analítico: \(x_k=\sqrt{2+\sqrt{3}}=\textbf{1.9318516525781366}\ldots\)
- Numérico (CGL + baricêntrico + Newton, N=30): **1.9318516525781364**
- Erro absoluto: **2.220446049250313e-16** (≈ piso de máquina)

---

## Observações rápidas

- **Convergência espectral** clara até a saturação numérica (\(N\sim 20{-}40\) já atinge \(10^{-13}\)/\(10^{-14}\)).
- Matrizes diferenciais de Chebyshev ficam **mal-condicionadas** para \(N\) grande (\(\kappa(D^{(1)})=\mathcal{O}(N^2)\)), o que explica a saturação.
- As figuras e tabelas geradas estão em `figures/` e `tables/` respectivamente.
