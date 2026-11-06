# Validation checklist

Run:

```bash
pytest -q
bibm-replicate verify-checkpoint /path/to/tabpfn-v2.6-classifier-v2.6_default.ckpt
bibm-replicate run-synthetic --config configs/synthetic_smoke.yaml
bibm-replicate analyze-synthetic --result-root results/synthetic_smoke
bibm-replicate generate-figures --experiment synthetic --result-root results/synthetic_smoke
```

Tests cover estimator cloning, parameter round trips, NumPy/DataFrame agreement, invalid inputs, exact deterministic KNN selection, discrete ancestry selection, identical continuum/DA-ICL bases, class-specific MST expansion, deterministic synthetic generation, metrics, checkpoint hashing, and figure creation. TabPFN inference itself is excluded from ordinary unit tests because it requires gated weights and a GPU; checkpoint integrity and the backend boundary are tested separately.

Run `python scripts/verify_repository.py` to confirm that the checksum manifest, the seven packaged Figure 1--4 panels, and the images embedded by the repository README agree exactly.

Before publishing a new run, verify that test IDs are absent from development IDs and context reports; every input block has aligned row counts; the exact checkpoint hash passes; random seeds/configs are saved; and regenerated figures point to the expected aggregate tables.
