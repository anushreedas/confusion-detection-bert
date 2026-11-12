import numpy as np
import pandas as pd
import pytest

from confusion_bert.data.splits import (
    assert_no_text_overlap,
    balance_by_downsampling,
    split_dataset,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    labels = rng.choice(4, size=n, p=[0.6, 0.2, 0.12, 0.08])
    return pd.DataFrame(
        {
            "sentence": [f"s{i}" for i in range(n)],
            "label": labels,
            "participant": rng.choice([str(i) for i in range(20)], size=n),
            "session": "1",
        }
    )


def test_stratified_split_sizes_and_proportions(df):
    tr, va, te = split_dataset(df, 0.1, 0.16, "stratified", seed=1)
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(len(te) / len(df) - 0.10) < 0.01
    assert abs(len(va) / len(df) - 0.144) < 0.01
    for part in (tr, va, te):
        assert abs((part.label == 3).mean() - (df.label == 3).mean()) < 0.02
    assert_no_text_overlap(train=tr, val=va, test=te)


def test_group_split_has_no_participant_overlap(df):
    tr, va, te = split_dataset(df, 0.2, 0.2, "group", seed=1)
    assert not set(tr.participant) & set(te.participant)
    assert not set(tr.participant) & set(va.participant)
    assert not set(va.participant) & set(te.participant)


def test_split_is_deterministic(df):
    a = split_dataset(df, seed=5)[2]
    b = split_dataset(df, seed=5)[2]
    pd.testing.assert_frame_equal(a, b)


def test_balance_equalises_classes_to_minimum(df):
    out = balance_by_downsampling(df, seed=1)
    counts = out.label.value_counts()
    assert counts.nunique() == 1
    assert counts.iloc[0] == df.label.value_counts().min()


def test_overlap_detected():
    a = pd.DataFrame({"sentence": ["x"], "label": [0]})
    with pytest.raises(AssertionError):
        assert_no_text_overlap(a=a, b=a)


def test_stratify_rare_class_error():
    d = pd.DataFrame(
        {"sentence": ["a", "b", "c"], "label": [0, 0, 1], "participant": "1", "session": "1"}
    )
    with pytest.raises(ValueError):
        split_dataset(d)
