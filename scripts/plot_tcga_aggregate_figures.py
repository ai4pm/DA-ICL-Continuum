#!/usr/bin/env python3
"""Regenerate redistributable TCGA plots from de-identified aggregate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TASKS = ("BRCA-PFI-Methylation", "KIRC-OS-MicroRNA", "UCEC-OS-MicroRNA")


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


def distance_plots(data_root: Path, output_root: Path):
    paths = []
    for metric in ("auroc", "nll"):
        data = pd.read_csv(data_root / f"tcga_afr_distance_{metric}_summary.csv")
        value = f"mean_{metric}"
        for task in TASKS:
            selected = data[data["task"].str.lower() == task.lower()].copy()
            if selected.empty:
                continue
            sns.set_theme(style="whitegrid", context="paper")
            figure, axis = plt.subplots(figsize=(5.4, 3.8))
            for method, group in selected.groupby("method", sort=False):
                group = group.sort_values("distance_bin_center")
                axis.plot(group["distance_bin_center"], group[value], marker="o", label=method)
            axis.set_xlabel("Genetic distance to AFR centroid")
            axis.set_ylabel(metric.upper())
            axis.set_title(task)
            axis.legend(frameon=False, fontsize=8)
            figure.tight_layout()
            destination = output_root / f"{task}_{metric}_by_distance.png"
            figure.savefig(destination, dpi=300)
            plt.close(figure)
            paths.append(destination)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/figure_data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/figures/tcga"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    produced = distance_plots(args.data_root, args.output_root)
    distribution = boxplot(args.data_root, args.output_root)
    if distribution is not None:
        produced.append(distribution)
    print("\n".join(map(str, produced)))


if __name__ == "__main__":
    main()

