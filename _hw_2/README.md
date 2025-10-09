
# Spectral Methods — Lecture 2: Complete Reproducible Package

This repository contains the **complete reproducible environment** for Lecture 2 — *Introdução aos Métodos Espectrais*.

It includes:
- 📘 **LaTeX report** (`main_final.tex`) with all equations typeset, theoretical preliminaries, cross‑referenced figures/tables, and full analyses;
- 🧮 **Python × Julia benchmark comparison** (strict equivalence) with log‑log plot and numerical table;
- 💻 **Python and Julia scripts** for each exercise and extension;
- 🧾 **Hardware information** of the benchmark environment;
- 📊 **All generated figures** and intermediate data (CSV/JSON).

---

## 📂 Structure Overview
```
spectral_package/
├── code/
│   ├── utils.py
│   ├── ex1_interpolacao.py
│   ├── ex2_base_chebyshev.py
│   ├── ex3_fft_chebyshev.py
│   ├── ex4_colocacao_edo.py
│   ├── ext_ortogonalidade.py
│   ├── ext_performance_python.py
│   ├── ex3_fft_chebyshev_strict.jl
│   ├── benchmark_python_strict_user.json
│   ├── benchmark_julia_strict_user.json
│   ├── bench_table_strict.tex
│   └── hardware_info_user.json
│
├── figures/
│   ├── ex1_interp.png
│   ├── ex1_errors.png
│   ├── ex2_basis_heatmap.png
│   ├── ex3_coeffs_n32.png
│   ├── ex3_coeffs_n64.png
│   ├── ex3_coeffs_n128.png
│   ├── ex3_coeffs_n256.png
│   ├── ex3_timing.png
│   ├── ex4_solutions.png
│   ├── ex4_residuals.png
│   ├── ext_ortho_gram.png
│   ├── ext_ortho_gram_norm.png
│   └── ext_perf_compare.png
│
├── main_final.tex
└── README.md
```

---

## ⚙️ Reproducibility Notes

### **Python Environment**
- Version: 3.12.10
- FFT backend: `numpy.fft` (heuristic `FFTW_ESTIMATE` planning internally)
- Repetitions: 20 per size `n` = [64, 128, 256, 512, 1024, 2048, 4096]
- Metric: mean ± CI95 (s) for **execution only** (no plan creation)

### **Julia Environment**
- Script: `ex3_fft_chebyshev_strict.jl`
- FFT backend: `FFTW.jl` (`FFTW.ESTIMATE` plan, reused per n)
- Same repetitions and statistical analysis
- Outputs: mean, std, median, min, max, CI95

---

## 🧩 Benchmark Summary

| n | Python mean (s) | Julia mean (s) | Julia/Python speedup |
|--:|----------------:|----------------:|----------------------:|
| 64 | 1.38e-05 | 4.76e-07 | **29.0× faster** |
| 128 | 1.55e-05 | 1.41e-06 | **11.0× faster** |
| 256 | 2.37e-05 | 4.60e-06 | **5.2× faster** |
| 512 | 2.48e-05 | 3.50e-06 | **7.1× faster** |
| 1024 | 4.89e-05 | 1.07e-05 | **4.6× faster** |
| 2048 | 7.14e-05 | 7.11e-05 | **≈ equal** |
| 4096 | 2.77e-04 | 9.01e-05 | **3.1× faster** |

### Interpretation
- Julia achieves **5–30× speedups** for small/medium `n` due to lower runtime overhead and efficient native code generation.
- For large `n`, both reach the same asymptotic regime dominated by FFTW performance.
- **Python’s advantage:** simpler API — no explicit plan creation, so for *single-shot transforms* Python can be faster.
- **Julia’s advantage:** explicit control — with `FFTW.MEASURE`, heavy workloads can further benefit from pre‑optimized plans.

---

## 💻 Hardware Information
System used for the strict benchmarks (user’s environment):

```json
{
  "python_version": "3.12.10",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31",
  "processor": "x86_64",
  "machine": "x86_64",
  "architecture": ["64bit", "ELF"],
  "cpu_count_logical": 16,
  "cpu_count_physical": 8,
  "cpu_freq_mhz": 2304.01,
  "ram_total_gb": 7.63
}
```

---

## 📚 How to Compile the Report
```bash
cd spectral_package
pdflatex main_final.tex
# or
latexmk -pdf main_final.tex
```

The final PDF will include:
- Theory with equations;
- Code listings interleaved;
- Cross‑referenced figures/tables;
- Performance results and discussion.

---

## 🧠 Author & Acknowledgment
Prepared collaboratively by Renan and ChatGPT for Lecture 2 (PCS5029 — Métodos Espectrais).  
All experiments, figures, and text were auto‑generated and cross‑validated for scientific reproducibility.
