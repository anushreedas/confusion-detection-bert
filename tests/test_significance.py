import numpy as np
import pytest

from confusion_bert.evaluation.metrics import accuracy, classification_report, macro_f1
from confusion_bert.evaluation.significance import (
    holm_adjust,
    make_stratified_subsamples,
    mcnemar_exact,
    paired_bootstrap,
    pairwise_compare,
    sign_test,
    subsample_scores,
)


def test_sign_test_all_wins_matches_report_value():
    # The capstone report lists 0.0000009 for 20/20 wins, i.e. 0.5**20.
    r = sign_test([1.0] * 20, [0.0] * 20)
    assert r.p_value == pytest.approx(0.5**20)
    assert (r.wins, r.losses, r.ties) == (20, 0, 0)


def test_sign_test_direction_and_ties():
    r = sign_test([1, 1, 0, 0, 0.5], [0, 0, 1, 1, 0.5])
    assert (r.wins, r.losses, r.ties) == (2, 2, 1)
    assert r.p_value > 0.5
    assert sign_test([1, 1], [1, 1]).p_value == 1.0


def test_mcnemar_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    a = y.copy()
    b = y.copy()
    flip = rng.random(500) < 0.3
    b[flip] = (b[flip] + 1) % 4
    r = mcnemar_exact(y, a, b)
    assert r.p_value < 1e-6 and r.b_only_correct == 0 < r.a_only_correct
    assert mcnemar_exact(y, b, a).p_value > 0.99


def test_bootstrap_identical_models_has_zero_diff():
    y = np.arange(200) % 4
    r = paired_bootstrap(y, y, y, n_resamples=200)
    assert r.mean_diff == 0 and r.ci_low == 0 == r.ci_high


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 1000)
    a = np.where(rng.random(1000) < 0.9, y, 0)
    b = np.where(rng.random(1000) < 0.7, y, 0)
    r = paired_bootstrap(y, a, b, n_resamples=500, seed=3)
    assert r.ci_low < r.mean_diff < r.ci_high and r.ci_low > 0 and r.p_value < 0.05


def test_holm_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == [0.5]


def test_subsamples_are_stratified_sized_and_seeded():
    y = np.repeat([0, 1, 2, 3], [600, 200, 120, 80])
    s1 = make_stratified_subsamples(y, 5, 0.3, seed=0)
    s2 = make_stratified_subsamples(y, 5, 0.3, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2, strict=True))
    for ix in s1:
        assert len(ix) == 300
        assert np.bincount(y[ix]).tolist() == [180, 60, 36, 24]
    assert not np.array_equal(s1[0], s1[1])


def test_metrics_against_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    assert accuracy(y, p) == pytest.approx(accuracy_score(y, p))
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))
    rep = classification_report(y, p)
    assert sum(c["support"] for c in rep["per_class"].values()) == 300


def test_pairwise_compare_table():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 4, 400)
    preds = {
        "good": np.where(rng.random(400) < 0.9, y, 1),
        "bad": np.where(rng.random(400) < 0.5, y, 1),
        "same": None,
    }
    preds["same"] = preds["good"].copy()
    df = pairwise_compare(y, preds, n_subsamples=10, bootstrap_resamples=100)
    assert len(df) == 6
    row = df[(df.model_a == "good") & (df.model_b == "bad")].iloc[0]
    assert row.sign_p < 0.01 and row.mcnemar_p < 0.01
    assert (df.sign_p_holm >= df.sign_p - 1e-12).all()
    scores = subsample_scores(y, preds["good"], make_stratified_subsamples(y, 3, 0.3))
    assert len(scores) == 3
