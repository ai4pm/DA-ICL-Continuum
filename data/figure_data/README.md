# Redistributable figure data

These CSV files contain only task-level or run-level aggregate metrics used to support the supplied TCGA figures:

- `tcga_core_methods_run_level_auroc.csv`: task/run AUROC for the four methods in the 18-task boxplot; no sample IDs or predictions.
- `tcga_icl_method_auroc_summary.csv`: task/method AUROC summaries for the ICL comparison.
- `tcga_afr_distance_auroc_summary.csv`: binned task/run AUROC summaries.
- `tcga_afr_distance_nll_summary.csv`: binned task/run NLL summaries.

The distance tables retain aggregate bin boundaries and counts needed to draw curves. They do not contain a patient's genotype distance, feature vector, outcome, prediction, or context membership. Run `python scripts/plot_tcga_aggregate_figures.py` from the repository root to generate redistributable plots under `results/figures/tcga/`.

