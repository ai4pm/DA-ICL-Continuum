# BIBM 2026 DA-ICL replication

This is the algorithm-focused replication repository for:

> Weizhi Li and Yan Cui. “Beyond Discrete Ancestry Groups: Distribution-Aware In-Context Learning for Cancer Outcome Prediction Across the Genetic Ancestry Continuum.” *2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*. IEEE, 2026.

The repository provides scikit-learn-style implementations of discrete ICL, ancestry-continuum ICL, and distribution-aware ICL (DA-ICL); a complete synthetic-data workflow; fixed configurations; tests; conventional baseline factories; and the exact manuscript figures. It intentionally does **not** acquire or preprocess TCGA. Users pass already prepared NumPy arrays or pandas DataFrames directly to the estimators.

## Quick start

```bash
cd /path/to/BIBM-replication
conda env create -f environment.yml
conda activate bibm-replication
python -m pip install -e .
pytest -q
```

Download the TabPFN-2.6 model card/weights after accepting any Prior Labs access terms, or point to an existing copy:

```bash
bibm-replicate fetch-checkpoint --destination checkpoints
bibm-replicate verify-checkpoint checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

The required SHA-256 is `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a`.

## Minimal array API

```python
from bibm_replication import DAICLClassifier

model = DAICLClassifier(
    n_neighbors=100,
    n_estimators=32,
    temperature=0.9,
    random_state=0,
    device="cuda",
    model_path="checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
)
model.fit(
    X_dev,
    y_dev,
    genotype_features=genotype_dev,
    dace_features=ae_dev,
    ancestry=ancestry_dev,       # retained for audits; not used to choose DA-ICL neighbors
    sample_ids=development_ids,  # optional; row indices are used when omitted
)
probability = model.predict_proba(
    X_test,
    genotype_features=genotype_test,
)[:, 1]
prediction = model.predict(X_test, genotype_features=genotype_test)
context_audit = model.last_context_report_
```

`fit()` validates and stores the development pool. Query-specific TabPFN contexts are constructed during `predict_proba()`; the foundation model is not target-task fine-tuned. All estimators expose `fit`, `predict`, `predict_proba`, `score`, `get_params`, `set_params`, `classes_`, and `n_features_in_`, and work with scikit-learn `clone`.

### Required arrays

| Input | Shape | Meaning |
|---|---:|---|
| `X_dev` | `(n_dev, 100)` | Development-patient omics PCA100 prediction features |
| `y_dev` | `(n_dev,)` | Binary development labels, encoded 0/1 |
| `genotype_dev` | `(n_dev, 30)` | Development-patient genotype PCA30 coordinates |
| `ae_dev` | `(n_dev, 20)` | Development-patient omics AE20 features for DACE |
| `ancestry_dev` | `(n_dev,)` | Reported ancestry, exactly `AFR` or `EUR` |
| `X_test` | `(n_test, 100)` | Query omics PCA100 prediction features |
| `genotype_test` | `(n_test, 30)` | Query genotype PCA30 coordinates |
| `ancestry_test` | `(n_test,)` | Query ancestry, exactly `AFR` or `EUR` |

- `DiscreteICLClassifier` requires `X`, `y`, and ancestry for development and query rows.
- `ContinuumICLClassifier` requires `X`, `y`, and genotype features.
- `DAICLClassifier` additionally requires development AE20 features; query AE20 features are not used.
- Every supplied block must have finite values and aligned rows. The development pool must contain both classes.

The fixed dimensions are deliberate paper defaults. They may be changed with `prediction_feature_dim`, `genotype_feature_dim`, and `dace_feature_dim` for unit testing or method adaptation, but that is no longer the reported setup. See [docs/input_data.md](docs/input_data.md).

### All three methods

```python
from bibm_replication import (
    DiscreteICLClassifier,
    ContinuumICLClassifier,
    DAICLClassifier,
)

discrete = DiscreteICLClassifier(model_path=checkpoint).fit(
    X_dev, y_dev, ancestry=ancestry_dev
)
p_discrete = discrete.predict_proba(X_test, ancestry=ancestry_test)[:, 1]

continuum = ContinuumICLClassifier(model_path=checkpoint).fit(
    X_dev, y_dev, genotype_features=genotype_dev
)
p_continuum = continuum.predict_proba(
    X_test, genotype_features=genotype_test
)[:, 1]

da_icl = DAICLClassifier(model_path=checkpoint).fit(
    X_dev, y_dev, genotype_features=genotype_dev, dace_features=ae_dev
)
p_da_icl = da_icl.predict_proba(
    X_test, genotype_features=genotype_test
)[:, 1]
```

Run the complete executable example with `python examples/array_api.py` on a CUDA node. The conventional factories are `make_elastic_net_classifier`, `make_random_forest_classifier`, and `make_xgboost_classifier`.

## Optional NPZ batch interface

The Python array API above is primary. For schedulers or shell pipelines, one archive may contain the same arrays under `X_dev`, `y_dev`, `genotype_dev`, `ae_dev`, `ancestry_dev`, `X_test`, `genotype_test`, and `ancestry_test`:

```bash
python examples/make_example_npz.py
bibm-replicate run-npz \
  --input examples/data/example_arrays.npz \
  --output results/example/da_icl.npz \
  --method da-icl \
  --model-path checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

The result contains probabilities and predictions; a neighboring `.contexts.json` records every base and DACE-added development member. If the archive contains an aligned `y_test` vector, the command also writes `.metrics.json` with AUROC, NLL, and Brier score. In Python, the same calculation is available as `bibm_replication.binary_metrics(y_test, probability)`. No multi-task CSV/Parquet format is required.

## Batch execution across tasks and seeds

Keep task-specific loading in your own authorized data layer and call the estimator with arrays:

```python
for task in tasks:
    for seed in range(30):
        arrays = load_prepared_arrays(task, seed)  # user-supplied function
        model = DAICLClassifier(random_state=seed, model_path=checkpoint)
        model.fit(
            arrays.X_dev, arrays.y_dev,
            genotype_features=arrays.genotype_dev,
            dace_features=arrays.ae_dev,
            sample_ids=arrays.development_ids,
        )
        probability = model.predict_proba(
            arrays.X_test, genotype_features=arrays.genotype_test
        )[:, 1]
        save_aggregate_or_authorized_output(task, seed, probability)
```

This repository does not define splits or labels and does not contain individual-level TCGA data. The caller is responsible for keeping test patients out of every development array and fitted transformation.

## Synthetic reproduction

The synthetic workflow generates its own data and is fully reproducible:

```bash
# Small end-to-end CPU validation
bibm-replicate run-synthetic --config configs/synthetic_smoke.yaml
bibm-replicate analyze-synthetic --result-root results/synthetic_smoke
bibm-replicate generate-figures --experiment synthetic --result-root results/synthetic_smoke

# Manuscript settings: 30 replicates and the complete rotation/offset sweeps
bibm-replicate run-synthetic --config configs/synthetic_paper.yaml
bibm-replicate analyze-synthetic --result-root results/synthetic
bibm-replicate generate-figures --experiment synthetic
```

The configuration uses 1,000 training and 600 test examples, three ancestry centroids, 35% admixed samples, two uniform omics variables, logistic scale 8, KNN100, deterministic replicate seeds, and the complete manuscript rotation and offset grids. Outputs include replicate metrics, aggregate summaries, five-bin genetic-distance metrics, and four regenerated figures. Full definitions are in [docs/synthetic.md](docs/synthetic.md).

## Manuscript figures and aggregate data

The 13 exact PNG files used by the manuscript are immutable references under [`figures/manuscript`](figures/manuscript); their hashes are in `SHA256SUMS`. Descriptions, captions, methods, populations, and source tables are catalogued in [docs/figures.md](docs/figures.md). Regenerated figures go under `results/.../figures` and never overwrite manuscript images. De-identified aggregate TCGA figure tables are under [`data/figure_data`](data/figure_data). No patient-level TCGA features, outcomes, predictions, context membership, or genetic distances are distributed.

## Environment, hardware, and runtime

The locked environment is Python 3.12.3, PyTorch 2.6.0/CUDA 12.4, TabPFN Python package 7.1.0 loading the TabPFN-2.6 checkpoint, scikit-learn 1.6.1, XGBoost 3.2.0, NumPy 2.1.2, pandas 2.3.1, SciPy 1.16.1, NetworkX 3.3, Matplotlib 3.10.5, and Seaborn 0.13.2.

The recommended replication profile is one NVIDIA H100 with at least 80 GB framebuffer per worker, 4 CPU cores, and 48 GB host RAM. The historical cluster also ran these small-context workloads on its V100S class, but the scheduler records did not preserve a portable per-device memory report, so users should verify available framebuffer with `nvidia-smi`. GPU memory and timing depend on query count, context expansion, and TabPFN caching. As planning ranges, a task/seed typically takes tens of minutes; synthetic smoke takes about 1–3 minutes on CPU, a 30-replicate synthetic sweep point about 3–10 minutes on 8 CPU cores, and the full sweep several CPU-hours serially. These are approximate, not performance guarantees. See [docs/environment.md](docs/environment.md).

## Repository map

```text
src/bibm_replication/     estimators, DACE, TabPFN backend, synthetic workflow, CLI
configs/                  immutable paper and smoke configurations
checkpoints/              weight acquisition and integrity instructions
examples/                 array API and optional NPZ examples
jobs/                     CPU synthetic and one-GPU NPZ Slurm templates
tests/                    interface, selection, DACE, determinism, and synthetic tests
docs/                     algorithms, data contract, figures, environment, validation
figures/manuscript/       exact manuscript reference images
data/figure_data/         de-identified aggregate figure tables
results/                  locally generated outputs (git-ignored)
```

## Scope and data access

Open-access TCGA omics may be obtained from the NCI Genomic Data Commons. Controlled germline genotype data must be requested by an authorized investigator through dbGaP and handled under its data-use agreement. This repository supplies the algorithmic interface only: no TCGA download, preprocessing, outcome construction, PCA/autoencoder fitting, or split generation is included. See [docs/input_data.md](docs/input_data.md).

## License and citation

Code is released under the [MIT License](LICENSE). Cite the paper using [CITATION.bib](CITATION.bib) or [CITATION.cff](CITATION.cff).

TabPFN-2.6 weights are distributed separately by Prior Labs through the [TabPFN-2.6 model card](https://huggingface.co/Prior-Labs/tabpfn_2_6) and are not relicensed or committed here.
