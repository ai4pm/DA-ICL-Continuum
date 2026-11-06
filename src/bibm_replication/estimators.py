"""Scikit-learn-style in-context classifiers used in the BIBM paper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ._validation import ancestries, binary_labels, identifiers, matrix
from .backend import predict_one
from .checkpoint import resolve_checkpoint
from .dace import class_specific_dace_indices


class _BaseICLClassifier(ClassifierMixin, BaseEstimator):
    """Common storage, validation, and per-query TabPFN prediction logic."""

    def __init__(
        self,
        *,
        n_neighbors: int = 100,
        n_estimators: int = 32,
        temperature: float = 0.9,
        random_state: int = 0,
        device: str = "cuda",
        model_path: str | None = None,
        backend: str = "tabpfn",
        prediction_feature_dim: int = 100,
        genotype_feature_dim: int = 30,
        dace_feature_dim: int = 20,
        scale_omics: bool = True,
        scale_genotype: bool = True,
        scale_dace: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.temperature = temperature
        self.random_state = random_state
        self.device = device
        self.model_path = model_path
        self.backend = backend
        self.prediction_feature_dim = prediction_feature_dim
        self.genotype_feature_dim = genotype_feature_dim
        self.dace_feature_dim = dace_feature_dim
        self.scale_omics = scale_omics
        self.scale_genotype = scale_genotype
        self.scale_dace = scale_dace

    @property
    def _needs_genotype(self) -> bool:
        return False

    @property
    def _needs_dace(self) -> bool:
        return False

    @property
    def _needs_ancestry(self) -> bool:
        return False

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        genotype_features: Any | None = None,
        dace_features: Any | None = None,
        ancestry: Any | None = None,
        sample_ids: Any | None = None,
    ):
        if int(self.n_neighbors) <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if float(self.temperature) <= 0:
            raise ValueError("temperature must be positive.")
        if self.backend not in {"tabpfn", "logistic"}:
            raise ValueError("backend must be 'tabpfn' or 'logistic'.")

        if hasattr(X, "columns") and all(isinstance(column, str) for column in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.X_dev_ = matrix(X, "X", columns=int(self.prediction_feature_dim))
        n_rows = len(self.X_dev_)
        self.y_dev_ = binary_labels(y, rows=n_rows)
        self.sample_ids_ = identifiers(sample_ids, rows=n_rows)
        self.ancestry_dev_ = (
            ancestries(ancestry, rows=n_rows) if self._needs_ancestry or ancestry is not None else None
        )
        self.genotype_dev_ = (
            matrix(
                genotype_features,
                "genotype_features",
                rows=n_rows,
                columns=int(self.genotype_feature_dim),
            )
            if self._needs_genotype or genotype_features is not None
            else None
        )
        self.dace_dev_ = (
            matrix(
                dace_features,
                "dace_features",
                rows=n_rows,
                columns=int(self.dace_feature_dim),
            )
            if self._needs_dace or dace_features is not None
            else None
        )
        if self._needs_genotype and int(self.n_neighbors) > n_rows:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the development-pool size {n_rows}."
            )

        self.model_path_ = self.model_path
        if self.backend == "tabpfn":
            self.model_path_ = str(resolve_checkpoint(self.model_path))

        self.genotype_scaler_ = None
        self.genotype_search_ = self.genotype_dev_
        if self.genotype_dev_ is not None and self.scale_genotype:
            self.genotype_scaler_ = StandardScaler().fit(self.genotype_dev_)
            self.genotype_search_ = self.genotype_scaler_.transform(self.genotype_dev_)
        self.dace_scaler_ = None
        self.dace_search_ = self.dace_dev_
        if self.dace_dev_ is not None and self.scale_dace:
            self.dace_scaler_ = StandardScaler().fit(self.dace_dev_)
            self.dace_search_ = self.dace_scaler_.transform(self.dace_dev_)

        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_features_in_ = self.X_dev_.shape[1]
        self.last_context_report_ = pd.DataFrame()
        return self

    def _base_indices(
        self,
        query_index: int,
        *,
        genotype_features: np.ndarray | None,
        ancestry: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _expanded_indices(self, base_indices: np.ndarray) -> np.ndarray:
        return np.empty(0, dtype=np.int64)

    def _query_inputs(
        self,
        X: Any,
        *,
        genotype_features: Any | None,
        ancestry: Any | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        check_is_fitted(self, attributes=["X_dev_", "y_dev_", "classes_"])
        query_x = matrix(X, "X", columns=self.n_features_in_)
        n_rows = len(query_x)
        query_genotype = None
        if self._needs_genotype:
            query_genotype = matrix(
                genotype_features,
                "genotype_features",
                rows=n_rows,
                columns=int(self.genotype_feature_dim),
            )
            if self.genotype_scaler_ is not None:
                query_genotype = self.genotype_scaler_.transform(query_genotype)
        query_ancestry = None
        if self._needs_ancestry:
            query_ancestry = ancestries(ancestry, rows=n_rows)
        elif ancestry is not None:
            query_ancestry = ancestries(ancestry, rows=n_rows)
        return query_x, query_genotype, query_ancestry

    def predict_proba(
        self,
        X: Any,
        *,
        genotype_features: Any | None = None,
        ancestry: Any | None = None,
    ) -> np.ndarray:
        query_x, query_genotype, query_ancestry = self._query_inputs(
            X, genotype_features=genotype_features, ancestry=ancestry
        )
        probabilities = np.empty((len(query_x), 2), dtype=float)
        report: list[dict[str, Any]] = []
        for query_index in range(len(query_x)):
            base = self._base_indices(
                query_index,
                genotype_features=query_genotype,
                ancestry=query_ancestry,
            )
            added = self._expanded_indices(base)
            context = np.concatenate((base, added))
            if len(np.unique(context)) != len(context):
                raise RuntimeError("Context construction produced duplicate development rows.")
            context_x = self.X_dev_[context]
            query_row = query_x[query_index : query_index + 1]
            if self.scale_omics:
                scaler = StandardScaler().fit(context_x)
                context_x = scaler.transform(context_x)
                query_row = scaler.transform(query_row)
            p_one = predict_one(
                context_x,
                self.y_dev_[context],
                query_row,
                backend=self.backend,
                n_estimators=int(self.n_estimators),
                temperature=float(self.temperature),
                random_state=int(self.random_state) + query_index,
                device=self.device,
                model_path=self.model_path_,
            )
            p_one = float(np.clip(p_one, 0.0, 1.0))
            probabilities[query_index] = (1.0 - p_one, p_one)
            report.append(
                {
                    "query_index": query_index,
                    "base_indices": base.tolist(),
                    "base_sample_ids": self.sample_ids_[base].tolist(),
                    "added_indices": added.tolist(),
                    "added_sample_ids": self.sample_ids_[added].tolist(),
                    "base_size": len(base),
                    "added_size": len(added),
                    "context_size": len(context),
                    "class_0_count": int(np.sum(self.y_dev_[context] == 0)),
                    "class_1_count": int(np.sum(self.y_dev_[context] == 1)),
                }
            )
        self.last_context_report_ = pd.DataFrame(report)
        return probabilities

    def predict(
        self,
        X: Any,
        *,
        genotype_features: Any | None = None,
        ancestry: Any | None = None,
    ) -> np.ndarray:
        return (self.predict_proba(X, genotype_features=genotype_features, ancestry=ancestry)[:, 1] >= 0.5).astype(int)

    def score(
        self,
        X: Any,
        y: Any,
        *,
        genotype_features: Any | None = None,
        ancestry: Any | None = None,
    ) -> float:
        prediction = self.predict(X, genotype_features=genotype_features, ancestry=ancestry)
        truth = np.asarray(y).reshape(-1)
        if len(truth) != len(prediction):
            raise ValueError("y must align with the query rows in X.")
        return float(accuracy_score(truth, prediction))


class DiscreteICLClassifier(_BaseICLClassifier):
    """Use all development patients matching each query's reported ancestry."""

    @property
    def _needs_ancestry(self) -> bool:
        return True

    def _base_indices(self, query_index, *, genotype_features, ancestry):
        selected = np.flatnonzero(self.ancestry_dev_ == ancestry[query_index])
        if len(selected) == 0:
            raise ValueError(f"No development patients have ancestry {ancestry[query_index]!r}.")
        return selected.astype(np.int64)


class ContinuumICLClassifier(_BaseICLClassifier):
    """Use each query's exact K nearest genetic neighbors from the combined pool."""

    @property
    def _needs_genotype(self) -> bool:
        return True

    def _base_indices(self, query_index, *, genotype_features, ancestry):
        delta = self.genotype_search_ - genotype_features[query_index]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        row_index = np.arange(len(squared_distance))
        order = np.lexsort((row_index, squared_distance))
        return order[: int(self.n_neighbors)].astype(np.int64)


class DAICLClassifier(ContinuumICLClassifier):
    """KNN-continuum ICL plus class-specific AE-space MST expansion."""

    @property
    def _needs_dace(self) -> bool:
        return True

    def _expanded_indices(self, base_indices: np.ndarray) -> np.ndarray:
        return class_specific_dace_indices(self.dace_search_, self.y_dev_, base_indices)
