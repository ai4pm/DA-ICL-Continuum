"""Public API for the BIBM DA-ICL replication package."""

from .baselines import (
    make_elastic_net_classifier,
    make_random_forest_classifier,
    make_xgboost_classifier,
)
from .estimators import ContinuumICLClassifier, DAICLClassifier, DiscreteICLClassifier
from .metrics import binary_metrics

__all__ = [
    "DiscreteICLClassifier",
    "ContinuumICLClassifier",
    "DAICLClassifier",
    "make_elastic_net_classifier",
    "make_random_forest_classifier",
    "make_xgboost_classifier",
    "binary_metrics",
]

__version__ = "1.0.0"
