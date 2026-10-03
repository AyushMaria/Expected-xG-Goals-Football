# Thesis: revised edition (2026)

LaTeX source of *Analysing Expected Goals (xG) in Football*, revised on all men's matches in StatsBomb's open data. The original 2022 thesis PDF (`Maria_349256_Thesis_MScCSTE.pdf` at the repository root) is unchanged.

- **Compiled PDF:** `Maria_349256_Thesis_MScCSTE_revised.pdf`
- **What changed:** see the preface in the PDF.

## Rebuild the figures

From the repository root, after `scripts/build_shots.py`, `train_evaluate.py`, `xg_tables.py` and `league_comparison.py` (see `docs/`):

```bash
python scripts/plot_eda.py --out thesis/images             # Methodology figures, results/eda_stats.json
python scripts/plot_thesis_results.py --out thesis/images  # Results figures
cp docs/figures/leagues.png thesis/images/
```

## Compile

```bash
cd thesis
latexmk -pdf main.tex      # needs biber / biblatex
```
