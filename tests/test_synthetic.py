from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bibm_replication.figures import generate_synthetic_figures
from bibm_replication.synthetic import SyntheticConfig, analyze_synthetic, generate_dataset, run_synthetic


def test_synthetic_generation_is_deterministic():
    config = SyntheticConfig(n_train=20, n_test=10, n_neighbors=5, n_replicates=1)
    first = generate_dataset(20, config, np.random.default_rng(123), 2.0, 0.5)
    second = generate_dataset(20, config, np.random.default_rng(123), 2.0, 0.5)
    for key in first:
        np.testing.assert_allclose(first[key], second[key])


def test_end_to_end_synthetic_smoke(tmp_path: Path):
    root = tmp_path / "result"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "result_root": str(root),
                "n_train": 60,
                "n_test": 40,
                "n_neighbors": 10,
                "n_replicates": 2,
                "local_oracle_size": 10,
                "rotation_values": [0.0, 10.0],
                "offset_values": [0.0, 1.0],
                "distance_rotation": 10.0,
                "distance_offset": 0.0,
            }
        ),
        encoding="utf-8",
    )
    run_synthetic(config_path)
    analyze_synthetic(root)
    figures = generate_synthetic_figures(root)
    assert len(figures) == 4
    assert all(path.is_file() for path in figures)
    metrics = pd.read_csv(root / "metrics_by_replicate.csv")
    assert set(metrics["model"]) == {
        "discrete_logistic",
        "empirical_continuum_logistic",
        "oracle_continuum_logistic",
        "oracle_probability",
    }
    assert metrics["replicate"].nunique() == 2

