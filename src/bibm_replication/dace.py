"""Distribution-aware context expansion used by DA-ICL."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def mst_crossing_candidates(candidates: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return candidate rows incident to candidate--base edges in a dense MST.

    This intentionally mirrors the manuscript implementation's deterministic
    Prim traversal. Candidate rows precede base rows, and ``numpy.argmin``
    resolves exact ties by row order.
    """
    candidates = np.asarray(candidates, dtype=float)
    base = np.asarray(base, dtype=float)
    n_candidates = len(candidates)
    if n_candidates == 0 or len(base) == 0:
        return np.empty(0, dtype=np.int64)

    combined = np.vstack((candidates, base))
    distances = pairwise_distances(combined, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    n_nodes = len(combined)
    in_tree = np.zeros(n_nodes, dtype=bool)
    parent = np.full(n_nodes, -1, dtype=np.int64)
    minimum = np.full(n_nodes, np.inf)
    in_tree[0] = True
    minimum[:] = distances[0]
    parent[:] = 0
    parent[0] = -1

    for _ in range(n_nodes - 1):
        node = int(np.argmin(np.where(in_tree, np.inf, minimum)))
        if np.isinf(minimum[node]):
            break
        in_tree[node] = True
        update = (~in_tree) & (distances[node] < minimum)
        minimum[update] = distances[node, update]
        parent[update] = node

    selected: set[int] = set()
    for child, source in enumerate(parent):
        if source < 0:
            continue
        source_candidate = source < n_candidates
        child_candidate = child < n_candidates
        if source_candidate != child_candidate:
            selected.add(int(source if source_candidate else child))
    return np.asarray(sorted(selected), dtype=np.int64)


def class_specific_dace_indices(
    features: np.ndarray,
    labels: np.ndarray,
    base_indices: np.ndarray,
) -> np.ndarray:
    """Select outside development rows using separate class-specific MSTs."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    outside_mask = np.ones(len(labels), dtype=bool)
    outside_mask[base_indices] = False
    outside_indices = np.flatnonzero(outside_mask)
    selected: list[int] = []
    for outcome in (0, 1):
        candidates = outside_indices[labels[outside_indices] == outcome]
        base = base_indices[labels[base_indices] == outcome]
        if len(candidates) == 0 or len(base) == 0:
            continue
        local = mst_crossing_candidates(features[candidates], features[base])
        selected.extend(candidates[local].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)

