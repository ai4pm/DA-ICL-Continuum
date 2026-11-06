"""Input validation shared by the public estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array


VALID_ANCESTRIES = frozenset({"AFR", "EUR"})


def matrix(value: Any, name: str, *, rows: int | None = None, columns: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required.")
    try:
        result = check_array(value, ensure_2d=True, dtype=np.float64, ensure_all_finite=True)
    except Exception as exc:
        raise ValueError(f"Invalid {name}: expected a finite numeric 2D array. {exc}") from exc
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} has {result.shape[0]} rows; expected {rows} to align with X.")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} has {result.shape[1]} columns; expected {columns}.")
    return result


def binary_labels(value: Any, *, rows: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        result = result.reshape(-1)
    if len(result) != rows:
        raise ValueError(f"y has {len(result)} rows; expected {rows} to align with X.")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise ValueError("y contains a missing or non-finite value.")
    try:
        result = result.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain only binary labels 0 and 1.") from exc
    values = np.unique(result)
    if not set(values.tolist()).issubset({0, 1}):
        raise ValueError(f"y must contain only binary labels 0 and 1; found {values.tolist()}.")
    if len(values) != 2:
        raise ValueError("The development pool must contain both outcome classes 0 and 1.")
    return result


def ancestries(value: Any, *, rows: int, name: str = "ancestry") -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required.")
    result = np.asarray(value, dtype=str).reshape(-1)
    if len(result) != rows:
        raise ValueError(f"{name} has {len(result)} rows; expected {rows} to align with X.")
    result = np.char.upper(np.char.strip(result))
    invalid = sorted(set(result.tolist()) - VALID_ANCESTRIES)
    if invalid:
        raise ValueError(f"{name} must contain only 'AFR' or 'EUR'; found {invalid}.")
    return result


def identifiers(value: Any, *, rows: int) -> np.ndarray:
    if value is None:
        return np.arange(rows, dtype=np.int64)
    result = np.asarray(value).reshape(-1)
    if len(result) != rows:
        raise ValueError(f"sample_ids has {len(result)} rows; expected {rows} to align with X.")
    if len(np.unique(result.astype(str))) != rows:
        raise ValueError("sample_ids must be unique within the development pool.")
    return result.copy()

