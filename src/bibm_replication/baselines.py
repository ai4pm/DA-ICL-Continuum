"""Conventional fixed-configuration baseline factories."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_elastic_net_classifier(random_state: int = 0, **overrides):
    parameters = dict(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=random_state,
    )
    parameters.update(overrides)
    return make_pipeline(StandardScaler(), LogisticRegression(**parameters))


def make_random_forest_classifier(random_state: int = 0, **overrides):
    parameters = dict(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=random_state)
    parameters.update(overrides)
    return RandomForestClassifier(**parameters)


def make_xgboost_classifier(random_state: int = 0, **overrides):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost==3.2.0 to use the XGBoost baseline.") from exc
    parameters = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    parameters.update(overrides)
    return XGBClassifier(**parameters)
