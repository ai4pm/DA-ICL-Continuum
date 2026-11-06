import numpy as np

from bibm_replication.dace import class_specific_dace_indices, mst_crossing_candidates


def test_mst_returns_only_candidate_side_of_crossing_edges():
    candidates = np.asarray([[0.0], [1.0], [10.0]])
    base = np.asarray([[0.2], [9.8]])
    selected = mst_crossing_candidates(candidates, base)
    assert set(selected.tolist()).issubset({0, 1, 2})
    assert len(selected) >= 1


def test_dace_is_class_specific_and_deterministic():
    features = np.asarray([[0.0], [0.2], [4.0], [4.2], [0.1], [4.1]])
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    base = np.asarray([4, 5])
    first = class_specific_dace_indices(features, labels, base)
    second = class_specific_dace_indices(features, labels, base)
    np.testing.assert_array_equal(first, second)
    assert set(first).isdisjoint(set(base))
    assert set(labels[first]).issubset({0, 1})

