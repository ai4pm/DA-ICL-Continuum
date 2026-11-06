import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from bibm_replication.metrics import binary_metrics


def test_metric_agreement():
    y = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.3, 0.7, 0.9])
    observed = binary_metrics(y, probability)
    assert observed["auroc"] == roc_auc_score(y, probability)
    assert observed["nll"] == log_loss(y, probability, labels=[0, 1])
