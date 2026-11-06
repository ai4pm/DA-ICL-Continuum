#!/usr/bin/env python3
"""Regenerate redistributable TCGA plots from de-identified aggregate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def boxplot(data_root: Path, output_root: Path):
    source = data_root / "tcga_core_methods_run_level_auroc.csv"
    if not source.is_file():
        return None
    data = pd.read_csv(source)
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=data, x="task_group", y="auroc", hue="method", ax=axis, fliersize=2)
    axis.tick_params(axis="x", rotation=70)
    axis.set_xlabel("")
    axis.set_ylabel("AUROC")
    axis.legend(frameon=False, ncol=4, fontsize=8)
    figure.tight_layout()
    destination = output_root / "tcga_core_methods_auroc_boxplot.png"
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/figure_data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/figures/tcga"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    distribution = boxplot(args.data_root, args.output_root)
    if distribution is None:
        raise FileNotFoundError(args.data_root / "tcga_core_methods_run_level_auroc.csv")
    print(distribution)


if __name__ == "__main__":
    main()
