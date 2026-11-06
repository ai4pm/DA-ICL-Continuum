"""End-to-end synthetic experiment described in the BIBM manuscript."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import binary_metrics, clipped_probability


ANCHORS = np.asarray([[0.15, 0.20], [0.85, 0.25], [0.50, 0.85]], dtype=float)


@dataclass(frozen=True)
class SyntheticConfig:
    result_root: str = "results/synthetic"
    n_train: int = 1000
    n_test: int = 600
    n_features: int = 2
    n_neighbors: int = 100
    n_replicates: int = 30
    seed: int = 0
    admixed_fraction: float = 0.35
    clustered_noise: float = 0.055
    admixed_noise: float = 0.035
    beta_shape: float = 1.4
    logistic_scale: float = 8.0
    local_oracle_size: int = 100
    rotation_values: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    offset_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
    distance_rotation: float = 10.0
    distance_offset: float = 0.0
    distance_bins: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SyntheticConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown synthetic configuration fields: {unknown}")
        for field in ("rotation_values", "offset_values"):
            if field in payload:
                payload[field] = tuple(float(value) for value in payload[field])
        config = cls(**payload)
        if config.n_features != 2:
            raise ValueError("The manuscript diagonal-boundary simulation requires n_features=2.")
        if not 0 <= config.admixed_fraction <= 1:
            raise ValueError("admixed_fraction must lie in [0, 1].")
        if config.n_neighbors > config.n_train:
            raise ValueError("n_neighbors cannot exceed n_train.")
        return config


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def sample_ancestry(n: int, config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    n_admixed = int(round(n * config.admixed_fraction))
    n_clustered = n - n_admixed
    labels = rng.integers(0, len(ANCHORS), size=n_clustered)
    clustered = ANCHORS[labels] + rng.normal(0.0, config.clustered_noise, size=(n_clustered, 2))
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]])
    pair_ids = rng.integers(0, len(pairs), size=n_admixed)
    interpolation = rng.beta(config.beta_shape, config.beta_shape, size=n_admixed)
    start = ANCHORS[pairs[pair_ids, 0]]
    end = ANCHORS[pairs[pair_ids, 1]]
    admixed = (1.0 - interpolation[:, None]) * start + interpolation[:, None] * end
    admixed += rng.normal(0.0, config.admixed_noise, size=(n_admixed, 2))
    ancestry = np.vstack((clustered, admixed))
    rng.shuffle(ancestry)
    return np.clip(ancestry, 0.0, 1.0)


def diagonal_margin(omics, ancestry, rotation: float, offset: float):
    theta = rotation * (ancestry[:, 0] - 0.5)
    base = np.asarray([-1.0, 1.0]) / np.sqrt(2.0)
    normal = np.column_stack(
        (
            np.cos(theta) * base[0] - np.sin(theta) * base[1],
            np.sin(theta) * base[0] + np.cos(theta) * base[1],
        )
    )
    return np.sum(omics * normal, axis=1) - offset * (ancestry[:, 1] - 0.5)


def generate_dataset(n, config, rng, rotation, offset):
    ancestry = sample_ancestry(n, config, rng)
    omics = rng.uniform(-1.0, 1.0, size=(n, 2))
    probability = sigmoid(config.logistic_scale * diagonal_margin(omics, ancestry, rotation, offset))
    labels = rng.binomial(1, probability).astype(int)
    return {"omics": omics, "ancestry": ancestry, "labels": labels, "probability": probability}


def _prevalence(labels):
    return float((np.sum(labels) + 0.5) / (len(labels) + 1.0))


def _logistic_predict(train_x, train_y, test_x):
    if len(np.unique(train_y)) < 2:
        return np.repeat(_prevalence(train_y), len(test_x))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6))
    model.fit(train_x, train_y)
    return clipped_probability(model.predict_proba(test_x)[:, 1])


def _nearest_anchor(ancestry):
    distance = np.linalg.norm(ancestry[:, None, :] - ANCHORS[None, :, :], axis=2)
    return np.argmin(distance, axis=1)


def predict_discrete(train, test):
    train_group = _nearest_anchor(train["ancestry"])
    test_group = _nearest_anchor(test["ancestry"])
    result = np.empty(len(test_group), dtype=float)
    for group in range(len(ANCHORS)):
        development = train_group == group
        queries = test_group == group
        result[queries] = _logistic_predict(train["omics"][development], train["labels"][development], test["omics"][queries])
    return result


def _neighbor_order(train_ancestry, query):
    distance = np.sum((train_ancestry - query) ** 2, axis=1)
    return np.lexsort((np.arange(len(distance)), distance))


def predict_empirical_continuum(train, test, k):
    result = np.empty(len(test["labels"]), dtype=float)
    for query_index, query in enumerate(test["ancestry"]):
        order = _neighbor_order(train["ancestry"], query)
        local_k = min(k, len(order))
        chosen = order[:local_k]
        while len(np.unique(train["labels"][chosen])) < 2 and local_k < len(order):
            local_k = min(len(order), max(local_k + 10, int(local_k * 1.25)))
            chosen = order[:local_k]
        result[query_index] = _logistic_predict(
            train["omics"][chosen], train["labels"][chosen], test["omics"][query_index : query_index + 1]
        )[0]
    return result


def predict_oracle_continuum(test, config, rotation, offset, rng):
    result = np.empty(len(test["labels"]), dtype=float)
    for query_index, query in enumerate(test["ancestry"]):
        local_x = rng.uniform(-1.0, 1.0, size=(config.local_oracle_size, 2))
        local_a = np.repeat(query[None, :], config.local_oracle_size, axis=0)
        local_p = sigmoid(config.logistic_scale * diagonal_margin(local_x, local_a, rotation, offset))
        local_y = rng.binomial(1, local_p).astype(int)
        result[query_index] = _logistic_predict(local_x, local_y, test["omics"][query_index : query_index + 1])[0]
    return result


def _settings(config):
    values = [(rotation, 0.0) for rotation in config.rotation_values]
    values.extend((0.0, offset) for offset in config.offset_values if offset != 0.0)
    return values


def _distance_rows(test, predictions, setting, replicate, n_bins):
    distance = np.min(np.linalg.norm(test["ancestry"][:, None, :] - ANCHORS[None, :, :], axis=2), axis=1)
    ordering = np.lexsort((np.arange(len(distance)), distance))
    bins = np.empty(len(distance), dtype=int)
    for bin_index, indices in enumerate(np.array_split(ordering, n_bins), start=1):
        bins[indices] = bin_index
    rows = []
    for bin_index in range(1, n_bins + 1):
        selected = bins == bin_index
        for model, probability in predictions.items():
            row = {
                "rotation": setting[0],
                "offset": setting[1],
                "replicate": replicate,
                "distance_bin": bin_index,
                "mean_distance": float(np.mean(distance[selected])),
                "n": int(np.sum(selected)),
                "events": int(np.sum(test["labels"][selected])),
                "model": model,
            }
            row.update(binary_metrics(test["labels"][selected], probability[selected]))
            rows.append(row)
    return rows


def run_synthetic(config_path: str | Path) -> Path:
    config = SyntheticConfig.from_yaml(config_path)
    root = Path(config.result_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    metric_rows = []
    distance_rows = []
    for rotation, offset in _settings(config):
        for replicate in range(config.n_replicates):
            seed = config.seed + replicate + int(round(rotation * 1000)) * 100_000 + int(round(offset * 1000)) * 10_000_000
            rng = np.random.default_rng(seed)
            train = generate_dataset(config.n_train, config, rng, rotation, offset)
            test = generate_dataset(config.n_test, config, rng, rotation, offset)
            predictions = {
                "discrete_logistic": predict_discrete(train, test),
                "empirical_continuum_logistic": predict_empirical_continuum(train, test, config.n_neighbors),
                "oracle_continuum_logistic": predict_oracle_continuum(
                    test, config, rotation, offset, np.random.default_rng(seed + 1_000_000_000)
                ),
                "oracle_probability": clipped_probability(test["probability"]),
            }
            for model, probability in predictions.items():
                row = {"rotation": rotation, "offset": offset, "replicate": replicate, "seed": seed, "model": model}
                row.update(binary_metrics(test["labels"], probability))
                metric_rows.append(row)
            if np.isclose(rotation, config.distance_rotation) and np.isclose(offset, config.distance_offset):
                distance_rows.extend(_distance_rows(test, predictions, (rotation, offset), replicate, config.distance_bins))
    pd.DataFrame(metric_rows).to_csv(root / "metrics_by_replicate.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(root / "metrics_by_genetic_distance.csv", index=False)
    return root


def analyze_synthetic(result_root: str | Path) -> Path:
    root = Path(result_root)
    metrics = pd.read_csv(root / "metrics_by_replicate.csv")
    group = metrics.groupby(["rotation", "offset", "model"], as_index=False)
    summary = group.agg(
        n_replicates=("replicate", "nunique"),
        auroc_mean=("auroc", "mean"),
        auroc_sem=("auroc", "sem"),
        nll_mean=("nll", "mean"),
        nll_sem=("nll", "sem"),
        brier_mean=("brier", "mean"),
        brier_sem=("brier", "sem"),
    )
    for metric in ("auroc", "nll", "brier"):
        summary[f"{metric}_ci95_low"] = summary[f"{metric}_mean"] - 1.96 * summary[f"{metric}_sem"]
        summary[f"{metric}_ci95_high"] = summary[f"{metric}_mean"] + 1.96 * summary[f"{metric}_sem"]
    summary.to_csv(root / "summary_metrics.csv", index=False)
    distance_path = root / "metrics_by_genetic_distance.csv"
    if distance_path.is_file():
        distance = pd.read_csv(distance_path)
        distance_summary = distance.groupby(["distance_bin", "model"], as_index=False).agg(
            n_replicates=("replicate", "nunique"),
            mean_distance=("mean_distance", "mean"),
            auroc_mean=("auroc", "mean"),
            auroc_sem=("auroc", "sem"),
            nll_mean=("nll", "mean"),
            nll_sem=("nll", "sem"),
        )
        distance_summary.to_csv(root / "summary_by_genetic_distance.csv", index=False)
    return root
