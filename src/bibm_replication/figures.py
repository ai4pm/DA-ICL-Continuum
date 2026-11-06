"""Figure generation from synthetic aggregate tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABELS = {
    "discrete_logistic": "Discrete logistic",
    "empirical_continuum_logistic": "Empirical-continuum logistic",
    "oracle_continuum_logistic": "Oracle-continuum logistic",
    "oracle_probability": "Oracle probability",
}


def _line(data, x, y, error, output, xlabel, ylabel):
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for model, group in data.groupby("model", sort=False):
        group = group.sort_values(x)
        axis.plot(group[x], group[y], marker="o", label=LABELS.get(model, model))
        axis.fill_between(group[x], group[y] - 1.96 * group[error], group[y] + 1.96 * group[error], alpha=0.15)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def generate_synthetic_figures(result_root: str | Path) -> list[Path]:
    root = Path(result_root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summary_metrics.csv")
    produced: list[Path] = []
    rotation = summary[summary["offset"].abs() < 1e-12]
    offset = summary[summary["rotation"].abs() < 1e-12]
    path = output / "auroc_vs_rotation.png"
    _line(rotation, "rotation", "auroc_mean", "auroc_sem", path, "Rotation parameter", "AUROC")
    produced.append(path)
    path = output / "auroc_vs_offset.png"
    _line(offset, "offset", "auroc_mean", "auroc_sem", path, "Offset parameter", "AUROC")
    produced.append(path)
    distance = pd.read_csv(root / "summary_by_genetic_distance.csv")
    for metric, ylabel in (("auroc", "AUROC"), ("nll", "Negative log loss")):
        path = output / f"{metric}_vs_genetic_distance.png"
        _line(
            distance,
            "mean_distance",
            f"{metric}_mean",
            f"{metric}_sem",
            path,
            "Distance to nearest ancestry centroid",
            ylabel,
        )
        produced.append(path)
    pd.DataFrame({"figure": [str(path.relative_to(root)) for path in produced]}).to_csv(
        output / "figure_index.csv", index=False
    )
    return produced

