import math

import numpy as np
import pytest

from confusion_bert.data.labels import (
    label_key,
    load_decoder,
    participant_and_session,
    to_seconds,
    token_labels,
)


def original_token_labels(change_points_decisec, timestamps):
    """Verbatim logic of the 2021 notebook (`get_labels`), used as an oracle."""
    to_frame = lambda x: [(math.ceil(t / 10), v) for (t, v) in x]  # noqa: E731
    time_state = sorted(set([(0, 0)] + to_frame(change_points_decisec)))  # noqa: RUF005
    return np.array(
        [max([s for s in time_state if t >= s[0]])[1] for t in timestamps], dtype=np.int64
    )


def test_matches_original_implementation_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_cp = int(rng.integers(0, 8))
        cps = [(int(rng.integers(0, 600)), int(rng.integers(0, 4))) for _ in range(n_cp)]
        ts = rng.uniform(0, 70, size=int(rng.integers(1, 40)))
        expected = original_token_labels(cps, ts)
        got = token_labels(ts, to_seconds(cps))
        np.testing.assert_array_equal(got, expected)


def test_before_first_change_point_is_not_confused():
    assert token_labels([0, 1, 2], [(5, 2)]).tolist() == [0, 0, 0]


def test_boundary_is_inclusive_and_latest_change_wins():
    assert token_labels([4, 5, 9, 10], [(5, 1), (10, 3)]).tolist() == [0, 1, 1, 3]


def test_same_timestamp_takes_larger_state():
    assert token_labels([5], [(5, 1), (5, 3)]).tolist() == [3]


def test_to_seconds_uses_ceil():
    assert to_seconds([(1, 1), (10, 2), (11, 3)]) == [(1, 1), (1, 2), (2, 3)]


def test_label_key_and_participant(tmp_path):
    p = tmp_path / "t2" / "903.csv"
    assert label_key(p) == "903t2"
    assert participant_and_session(p) == ("903", "2")


def test_load_decoder_errors():
    with pytest.raises(ValueError):
        load_decoder("nocolon")
    assert callable(load_decoder("tests.helpers:fake_decoder"))
