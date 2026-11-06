"""Minimal array-based examples for all three classifiers."""

import numpy as np

from bibm_replication import ContinuumICLClassifier, DAICLClassifier, DiscreteICLClassifier

rng = np.random.default_rng(0)
n_dev, n_test = 140, 4
X_dev = rng.normal(size=(n_dev, 100))
y_dev = np.tile([0, 1], n_dev // 2)
genotype_dev = rng.normal(size=(n_dev, 30))
ae_dev = rng.normal(size=(n_dev, 20))
ancestry_dev = np.where(np.arange(n_dev) % 3, "EUR", "AFR")
X_test = rng.normal(size=(n_test, 100))
genotype_test = rng.normal(size=(n_test, 30))
ancestry_test = np.asarray(["EUR", "AFR", "EUR", "AFR"])

discrete = DiscreteICLClassifier(device="cuda").fit(X_dev, y_dev, ancestry=ancestry_dev)
print(discrete.predict_proba(X_test, ancestry=ancestry_test)[:, 1])

continuum = ContinuumICLClassifier(device="cuda").fit(
    X_dev, y_dev, genotype_features=genotype_dev
)
print(continuum.predict_proba(X_test, genotype_features=genotype_test)[:, 1])

da_icl = DAICLClassifier(device="cuda").fit(
    X_dev,
    y_dev,
    genotype_features=genotype_dev,
    dace_features=ae_dev,
    ancestry=ancestry_dev,
)
print(da_icl.predict_proba(X_test, genotype_features=genotype_test)[:, 1])
print(da_icl.last_context_report_[["base_size", "added_size", "context_size"]])

