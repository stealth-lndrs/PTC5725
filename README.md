# PCS5029 – Métodos Espectrais · Relatórios e Códigos

> **Autor:** Renan de Luca Avila  
> **Disciplina:** *PCS5029 – Introdução aos Métodos Espectrais (EPUSP)*  
> **Escopo:** Relatórios em LaTeX + código Python reprodutível para interpolação, matrizes de diferenciação e método baricêntrico (e demais tópicos ao longo do curso).

Última data de atualização: 01/10/2025.

---

## 🌳 Estrutura do repositório

```
PTC5725/
├─ .gitignore
├─ LICENSE                  # Apache-2.0, igual ao repo de referência
├─ README.md                # README (PT-BR) que já escrevemos
├─ _hw1_1/                  # Lista 1 (Q1–Q2) — relatório e figuras
│  ├─ main.tex              # código-fonte LaTeX com listagens inline
│  ├─ references.bib
│  └─ figs/
│     ├─ q1_exp-x.png
│     ├─ q1_sinpi_x.png
│     ├─ q1_cospi_x.png
│     ├─ q1_poly_6.png
│     ├─ fig_equidistante.png
│     └─ fig_chebyshev.png
└─ _hw_1_2/                 # Lista 1 (Q3–Q4) — relatório e figuras
   ├─ main.tex              # inclui D, baricêntrica e passos intermediários
   ├─ references.bib
   └─ figs/
      ├─ q3_deriv_eq.png
      ├─ q3_deriv_ch.png
      ├─ q3_erro_eq.png
      ├─ q3_erro_ch.png
      └─ q4_bary_cheb.png

```

---

## 🚀 Primeiros passos

### 1) Ambiente Python

```bash
# Criar & ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -U pip
pip install numpy matplotlib
```

### 2) Regenerar figuras

Execute os scripts abaixo; eles salvarão PNGs na pasta de figuras do relatório correspondente:

```bash
# Na raiz do repositório
python code/make_figs_q1.py   # Interpolação (Lagrange): exp(-x), sin(pi x), cos(pi x), polinômio
python code/make_figs_q3.py   # Matriz de diferenciação: f'(x) vs Df
python code/make_figs_q4.py   # Interpolação baricêntrica (Chebyshev-Lobatto)
```
---

## 🧩 Módulos principais (guia rápido)

### `code/lagrange_interpolation.py`
- **O que faz:** avalia o polinômio interpolador de Lagrange \( P_n(x) = \sum_j f(x_j)\,\ell_j(x) \).
- **Uso:**
  ```python
  from lagrange_interpolation import lagrange_interpolation
  yk = lagrange_interpolation(x_nodes, y_nodes, x_eval)
  ```

### `code/diff_matrix.py`
- **O que faz:** constrói a **matriz de diferenciação** \( D \) em grade arbitrária usando pesos baricêntricos \(w_j = 1/\prod_{m\ne j}(x_j-x_m)\), com
  \[
    D_{ij}=\frac{w_j}{w_i}\frac{1}{x_i-x_j}\ (i\ne j),\quad
    D_{ii}=-\sum_{j\ne i}D_{ij}.
  \]
- **Uso:**
  ```python
  from diff_matrix import differentiate
  fprime_at_nodes = differentiate(x_nodes, f(x_nodes), k=1)
  ```

### `code/barycentric.py`
- **O que faz:** avaliação estável por **interpolação baricêntrica**; inclui pesos fechados para Chebyshev-Lobatto.
- **Uso:**
  ```python
  from barycentric import bary_weights_cheb2, bary_eval
  w = bary_weights_cheb2(n)      # x_j = cos(j*pi/n)
  yq = bary_eval(x_nodes, y_nodes, x_query, w=w)
  ```

---

## 📈 Figuras reprodutíveis

- **Q1** (Interpolação):  
  `make_figs_q1.py` gera figuras com **função original** vs **interpolação de Lagrange** usando nós **equidistantes**, além do comparativo clássico Runge vs Chebyshev.  
  As figuras são referenciadas no LaTeX por labels do tipo `\ref{fig:q1-exp}`.

- **Q3** (Matriz de diferenciação):  
  `make_figs_q3.py` plota derivada exata vs numérica e perfis de erro para \( f(x) = e^x \sin(5x) \), em grades **equidistantes** vs **Chebyshev-Lobatto**.

- **Q4** (Baricêntrica):  
  `make_figs_q4.py` plota interpolação baricêntrica de \( f(x) = 1/(1+16x^2) \) em nós de Chebyshev-Lobatto.

> Ajuste funções e número de nós nos scripts e recompile o relatório conforme necessário.

---

## 🧪 Checks rápidos (sugestões)

- `bary_eval(x, y, x)` deve retornar exatamente `y` (propriedade de interpolação).  
- As linhas de \( D \) devem somar zero: `np.allclose(D.sum(axis=1), 0)` (derivada de constante é zero).  
- Para polinômios de grau ≤ \(n\), avaliações Lagrange/baricêntrica devem ser exatas (até *round-off*).

---

## 🗂 Convenções

- **Relatórios:** `reports/hwXX/main.tex`, `figs/`, `references.bib`.  
- **Figuras:** `snake_case` com prefixos curtos, ex.: `q1_exp-x.png`, `q3_deriv_ch.png`.  
- **Código:** módulos autocontidos, PEP-8 quando possível.

---

## 🧾 Reconhecimento de uso de LLM

Este repositório eventualmente usa um LLM para:
- rascunho de trechos em LaTeX,
- geração de *boilerplate* Python,
- e sugestões de figuras/validações.

**Todo código e texto são revisados e validados pelo autor** antes da submissão.

---

### TL;DR

```bash
# 1) Ambiente
python -m venv .venv && source .venv/bin/activate
pip install -U pip numpy matplotlib

# 2) Figuras
python code/make_figs_q1.py
python code/make_figs_q3.py
python code/make_figs_q4.py

# 3) PDF
cd reports/hw01 && latexmk -pdf main.tex
```
