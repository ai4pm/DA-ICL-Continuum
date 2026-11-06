"""Prediction backends for one query-specific context."""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def constant_probability(y: np.ndarray) -> float:
    """Return the observed class for a single-class context."""
    return float(np.mean(y))


def _class_one_probability(model: object, row: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(row), dtype=float)
    classes = np.asarray(model.classes_).reshape(-1)
    if len(classes) == 1:
        return float(classes[0] == 1)
    column = int(np.flatnonzero(classes == 1)[0])
    return float(probabilities[0, column])


def predict_one(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    *,
    backend: str,
    n_estimators: int,
    temperature: float,
    random_state: int,
    device: str,
    model_path: str | None,
) -> float:
    if len(np.unique(context_y)) < 2:
        return constant_probability(context_y)
    if backend == "logistic":
        model = LogisticRegression(max_iter=2000, random_state=random_state)
        model.fit(context_x, context_y)
        return _class_one_probability(model, query_x)
    if backend != "tabpfn":
        raise ValueError("backend must be 'tabpfn' or 'logistic'.")

    try:
        import torch
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "TabPFN prediction requires torch and tabpfn. Install requirements.txt, "
            "or use backend='logistic' only for interface tests."
        ) from exc

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    requested = {
        "n_estimators": int(n_estimators),
        "softmax_temperature": float(temperature),
        "random_state": int(random_state),
        "device": device,
    }
    if model_path is not None:
        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TabPFN checkpoint does not exist: {checkpoint}")
        requested["model_path"] = str(checkpoint)
    parameters = inspect.signature(TabPFNClassifier).parameters
    kwargs = {name: value for name, value in requested.items() if name in parameters}
    model = TabPFNClassifier(**kwargs)
    model.fit(context_x, context_y)
    return _class_one_probability(model, query_x)
