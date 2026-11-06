# Redistributable figure data

This directory contains the de-identified run-level metrics supporting Figure 4:

- `tcga_core_methods_run_level_auroc.csv`: task/run AUROC for the four methods in the 18-task boxplot; no sample IDs or predictions.

The tables do not contain patient identifiers, feature vectors, outcomes, predictions, context membership, or genotype distances. Run `python scripts/plot_tcga_aggregate_figures.py` from the repository root to regenerate the Figure 4 boxplot under `results/figures/tcga/`.
