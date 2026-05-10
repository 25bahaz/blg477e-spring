# Building the progress report

There are two source files in this folder:

## 1. `progress_report.tex` (preferred — IEEE conference style)

Requires the **IEEEtran** document class.  Most standard LaTeX
distributions ship it; install with one of:

```bash
# Debian / Ubuntu
sudo apt-get install texlive-publishers

# Fedora
sudo dnf install texlive-IEEEtran

# Windows / macOS – install the "ieeetran" package via tlmgr / MikTeX manager
tlmgr install ieeetran
```

Then build:

```bash
pdflatex progress_report.tex
pdflatex progress_report.tex     # second pass for cross-refs
```

If you use [Overleaf](https://overleaf.com), `IEEEtran.cls` is already
available — just upload the project folder.

## 2. `check.tex`  (fallback — vanilla `article` class)

Identical content, but uses the standard `article` class with a
two-column layout, so it compiles on **any** texlive distribution
without installing extra packages.  The pre-built PDF
(`progress_report.pdf`) was generated from this fallback.

To regenerate it:

```bash
python3 _make_check.py
pdflatex check.tex
pdflatex check.tex
```

## Figures

The figures referenced from `..//figures/` are produced by the project's
demo command:

```bash
cd ..
python -m src.main demo
python src/make_compression_chart.py outputs/compression.json figures/compression_chart.png
```
