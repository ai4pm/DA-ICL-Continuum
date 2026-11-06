"""Binary evaluation metrics used by the replication workflows."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


EPSILON = 1e-6


def clipped_probability(value):
    return np.clip(np.asarray(value, dtype=float), EPSILON, 1.0 - EPSILON)


def binary_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = clipped_probability(probability)
    return {
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
        "nll": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
    }

