# PTC5725 – Tarefa 04: Métodos Espectrais e Método de Newton

Este repositório contém os códigos, resultados e relatório em LaTeX referentes à **Tarefa 04** da disciplina **PTC5725 – Introdução aos Métodos Espectrais (Escola Politécnica da USP)**.  
O projeto apresenta aplicações do **método de Newton** (em versões clássica e amortecida) e da **colocalização de Chebyshev** em diferentes contextos de equações não lineares.

---

## 📘 Estrutura dos Exercícios

| Exercício | Descrição | Método Principal |
|------------|------------|------------------|
| **1** | Resolução de EDO não linear \\( y'' = e^y, \\, y(\\pm1)=1 \\) | Colocalização de Chebyshev + Newton-Raphson |
| **2** | Sistema 2D não linear \\( f_1(x,y), f_2(x,y) \\) | Newton–Jacobian (puro e amortecido) + comparação com `fsolve` |
| **3** | Sistema 3D acoplado \\( f_1(x,y,z), f_2(x,y,z), f_3(x,y,z) \\) | Newton–Jacobian 3D (puro e amortecido) + comparação com `fsolve` |

---

## 🧠 Metodologia Resumida

1. **Colocalização de Chebyshev (Ex. 1):**
   - Construção da matriz diferencial \\( D \\) e \\( D^2 \\);
   - Resolução iterativa do sistema não linear \\( F(y)=D^2y - e^y = 0 \\);
   - Cálculo do resíduo \\( R(x)=y''-e^y \\) e do decaimento espectral de \\(|c_k|\\).

2. **Método de Newton–Jacobian (Ex. 2 e 3):**
   - Linearização local \\( J(x_k)\\Delta x = -F(x_k) \\);
   - Atualização \\( x_{k+1} = x_k + \\alpha\\Delta x \\);
   - Versões **pura** (\\(\\alpha=1\\)) e **amortecida** (com backtracking).

3. **Comparação com `fsolve`:**
   - Validação numérica e análise de robustez;
   - `fsolve` combina Newton e métodos de região de confiança (Levenberg–Marquardt / dogleg).

---

## 🧩 Estrutura do Projeto

```
PTC5725_Tarefa04/
├── code/
│   ├── ex1_cheb_newton.py
│   ├── ex2_newton_vs_fsolve.py
│   └── ex3_newton3d_vs_fsolve.py
├── figures/
│   ├── ex1_*.pdf
│   ├── ex2_*.pdf
│   └── ex3_*.pdf
├── tables/
│   ├── ex1_*.csv / .json
│   ├── ex2_*.csv
│   └── ex3_*.csv
├── refs/
│   └── refs.bib
└── main.tex
```

O relatório em LaTeX (`main.tex`) contém:
- Resumo e Enunciados;
- Resolução e análise detalhada de cada exercício;
- Glossário e Instruções de setup;
- Códigos completos importados via `\\inputminted`.

---

## ⚙️ Ambiente e Dependências

### Versões utilizadas
- **Python:** 3.12.10  
- **Bibliotecas:**
  - `numpy==1.26.0`
  - `scipy==1.14.1`
  - `pandas==2.2.2`
  - `matplotlib==3.9.2`
  - (para LaTeX) `minted==2.9`, `Pygments==2.18.0`

### Instalação do ambiente
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install numpy scipy pandas matplotlib
```

### Execução dos scripts
```bash
python code/ex1_cheb_newton.py
python code/ex2_newton_vs_fsolve.py
python code/ex3_newton3d_vs_fsolve.py
```

As figuras e tabelas serão geradas automaticamente nas pastas correspondentes.

---

## 🧮 Compilação do Relatório

O relatório principal está em **LaTeX** e utiliza o pacote `minted` para exibir código com sintaxe colorida.

Para compilar (necessário `pdflatex` com permissão de shell):

```bash
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

---

## 📈 Resultados Principais

- **Ex. 1:** convergência espectral evidenciada pelo decaimento exponencial de \\(|c_k|\\).
- **Ex. 2:** robustez do método amortecido e validação com `fsolve`.
- **Ex. 3:** convergência estável do Newton 3D com amortecimento (trajetória suave e \\(\\|F\\|_2\\) decrescente).

---

## 👨‍💻 Autor
**Renan — Escola Politécnica da USP**  
Disciplina **PTC5725 – Introdução aos Métodos Espectrais**  
Professor **Osvaldo Guimarães**

---

## 📜 Licença
Este material é disponibilizado para fins acadêmicos e reprodutibilidade científica.  
Sinta-se à vontade para clonar, executar e expandir os códigos para novos experimentos.
