"""Create a synthetic archive solely to demonstrate the optional NPZ CLI."""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
n_dev, n_test = 140, 8
Path("examples/data").mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    "examples/data/example_arrays.npz",
    X_dev=rng.normal(size=(n_dev, 100)),
    y_dev=np.tile([0, 1], n_dev // 2),
    genotype_dev=rng.normal(size=(n_dev, 30)),
    ae_dev=rng.normal(size=(n_dev, 20)),
    ancestry_dev=np.where(np.arange(n_dev) % 3, "EUR", "AFR"),
    sample_ids=np.asarray([f"development_{index:04d}" for index in range(n_dev)]),
    X_test=rng.normal(size=(n_test, 100)),
    genotype_test=rng.normal(size=(n_test, 30)),
    ancestry_test=np.where(np.arange(n_test) % 2, "EUR", "AFR"),
)
