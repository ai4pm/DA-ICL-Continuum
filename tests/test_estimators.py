import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from bibm_replication import ContinuumICLClassifier, DAICLClassifier, DiscreteICLClassifier


COMMON = dict(
    backend="logistic",
    device="cpu",
    prediction_feature_dim=4,
    genotype_feature_dim=2,
    dace_feature_dim=3,
    n_neighbors=4,
    random_state=11,
)


def test_clone_and_parameter_round_trip():
    original = DAICLClassifier(**COMMON)
    copied = clone(original)
    assert copied.get_params() == original.get_params()
    copied.set_params(n_neighbors=3)
    assert copied.n_neighbors == 3


def test_discrete_contexts_are_same_ancestry(small_arrays):
    values = small_arrays
    model = DiscreteICLClassifier(**COMMON).fit(
        values["X"], values["y"], ancestry=values["ancestry"], sample_ids=values["ids"]
    )
    probability = model.predict_proba(values["X_test"], ancestry=values["ancestry_test"])
    assert probability.shape == (3, 2)
    assert model.classes_.tolist() == [0, 1]
    assert model.n_features_in_ == 4
    assert model.last_context_report_.iloc[0].base_indices == [0, 1, 2, 3, 4]
    assert model.last_context_report_.iloc[1].base_indices == [5, 6, 7, 8, 9, 10, 11]
    assert model.last_context_report_.iloc[0].added_size == 0


def test_exact_knn_and_deterministic_ties():
    X = np.column_stack((np.arange(6), np.arange(6), np.arange(6), np.arange(6))).astype(float)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    genotype = np.asarray([[0, 0], [0, 0], [1, 0], [-1, 0], [2, 0], [-2, 0]], dtype=float)
    model = ContinuumICLClassifier(**{**COMMON, "n_neighbors": 4}).fit(X, y, genotype_features=genotype)
    model.predict_proba(X[:1], genotype_features=np.asarray([[0, 0]], dtype=float))
    assert model.last_context_report_.iloc[0].base_indices == [0, 1, 2, 3]


def test_da_and_continuum_have_identical_base(small_arrays):
    values = small_arrays
    continuum = ContinuumICLClassifier(**COMMON).fit(
        values["X"], values["y"], genotype_features=values["genotype"], sample_ids=values["ids"]
    )
    da_icl = DAICLClassifier(**COMMON).fit(
        values["X"],
        values["y"],
        genotype_features=values["genotype"],
        dace_features=values["dace"],
        sample_ids=values["ids"],
    )
    continuum.predict_proba(values["X_test"], genotype_features=values["genotype_test"])
    da_icl.predict_proba(values["X_test"], genotype_features=values["genotype_test"])
    assert continuum.last_context_report_.base_indices.tolist() == da_icl.last_context_report_.base_indices.tolist()
    for row in da_icl.last_context_report_.itertuples():
        assert set(row.base_indices).isdisjoint(row.added_indices)


def test_numpy_and_dataframe_equivalence(small_arrays):
    values = small_arrays
    numpy_model = ContinuumICLClassifier(**COMMON).fit(
        values["X"], values["y"], genotype_features=values["genotype"]
    )
    frame_model = ContinuumICLClassifier(**COMMON).fit(
        pd.DataFrame(values["X"], columns=list("abcd")),
        pd.Series(values["y"]),
        genotype_features=pd.DataFrame(values["genotype"]),
    )
    numpy_probability = numpy_model.predict_proba(values["X_test"], genotype_features=values["genotype_test"])
    frame_probability = frame_model.predict_proba(
        pd.DataFrame(values["X_test"], columns=list("abcd")),
        genotype_features=pd.DataFrame(values["genotype_test"]),
    )
    np.testing.assert_allclose(numpy_probability, frame_probability)


def test_clear_validation_errors(small_arrays):
    values = small_arrays
    with pytest.raises(ValueError, match="both outcome classes"):
        ContinuumICLClassifier(**COMMON).fit(
            values["X"], np.zeros(len(values["y"])), genotype_features=values["genotype"]
        )
    with pytest.raises(ValueError, match="align with X"):
        DiscreteICLClassifier(**COMMON).fit(
            values["X"], values["y"], ancestry=values["ancestry"][:-1]
        )
    with pytest.raises(ValueError, match="AFR.*EUR"):
        DiscreteICLClassifier(**COMMON).fit(
            values["X"], values["y"], ancestry=np.asarray(["OTHER"] * len(values["y"]))
        )
    with pytest.raises(ValueError, match="exceeds"):
        ContinuumICLClassifier(**{**COMMON, "n_neighbors": 100}).fit(
            values["X"], values["y"], genotype_features=values["genotype"]
        )

