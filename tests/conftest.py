import numpy as np
import pytest


@pytest.fixture
def small_arrays():
    rng = np.random.default_rng(7)
    n_dev = 12
    return {
        "X": rng.normal(size=(n_dev, 4)),
        "y": np.asarray([0, 1] * 6),
        "genotype": np.asarray([[index // 2, index % 2] for index in range(n_dev)], dtype=float),
        "dace": rng.normal(size=(n_dev, 3)),
        "ancestry": np.asarray(["AFR"] * 5 + ["EUR"] * 7),
        "ids": np.asarray([f"id_{index}" for index in range(n_dev)]),
        "X_test": rng.normal(size=(3, 4)),
        "genotype_test": np.asarray([[0.0, 0.0], [2.5, 0.5], [5.0, 1.0]]),
        "ancestry_test": np.asarray(["AFR", "EUR", "EUR"]),
    }

