import json

import pandas as pd

from confusion_bert.data.labels import load_decoder
from confusion_bert.data.sentences import build_sentence_table, read_transcript, segment


def test_segment_max_label_and_terminators():
    words = ["a", "b", ".", "c", "?", "d"]
    labels = [0, 2, 0, 1, 0, 3]
    assert segment(words, labels) == [("a b", 2), ("c", 1)]
    assert segment(words, labels, keep_trailing_fragment=True)[-1] == ("d", 3)


def test_segment_skips_empty_sentences_and_empty_tokens():
    assert segment(["a", ".", ".", "", "b", "?"], [1, 0, 0, 0, 2, 0]) == [("a", 1), ("b", 2)]


def test_read_transcript_handles_missing_values(tmp_path):
    f = tmp_path / "t1" / "1.csv"
    f.parent.mkdir()
    pd.DataFrame({"word": ["hi", None, "yo"], "start": [1.0, 5.0, None], "end": [2, 6, 7]}).to_csv(
        f, index=False
    )
    words, times = read_transcript(f)
    assert words == ["hi", "", "yo"]
    assert times.tolist() == [1.0, 0.0, 0.0]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["word", "start", "end"]).to_csv(path, index=False)


def test_duplicate_sentence_does_not_merge_into_next_regression(tmp_path):
    """The original code glued a duplicate sentence onto the following one."""
    raw = tmp_path / "raw"
    _write(
        raw / "t1" / "1.csv",
        [("ok", 1, 1), (".", 1, 1), ("ok", 2, 2), (".", 2, 2), ("next", 3, 3), (".", 3, 3)],
    )
    table = {"1t1": "0:0"}
    df = build_sentence_table(raw, table, load_decoder("tests.helpers:fake_decoder"))
    assert df["sentence"].tolist() == ["ok", "next"]


def test_global_dedup_across_participants_keeps_first(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "t1" / "1.csv", [("hello", 1, 1), (".", 1, 1)])
    _write(raw / "t1" / "2.csv", [("hello", 1, 1), (".", 1, 1), ("bye", 2, 2), (".", 2, 2)])
    df = build_sentence_table(
        raw, {"1t1": "0:0", "2t1": "0:0"}, load_decoder("tests.helpers:fake_decoder")
    )
    assert df[["sentence", "participant"]].values.tolist() == [["hello", "1"], ["bye", "2"]]


def test_labels_applied_from_decoder(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "t1" / "1.csv", [("a", 1, 1), (".", 1, 1), ("b", 30, 30), (".", 30, 30)])
    df = build_sentence_table(raw, {"1t1": "100:2"}, load_decoder("tests.helpers:fake_decoder"))
    assert df["label"].tolist() == [0, 2]  # change at 100 ds = 10 s


def test_missing_annotation_raises(tmp_path):
    import pytest

    raw = tmp_path / "raw"
    _write(raw / "t1" / "1.csv", [("a", 1, 1), (".", 1, 1)])
    with pytest.raises(KeyError):
        build_sentence_table(raw, {}, load_decoder("tests.helpers:fake_decoder"))
    json.dumps({})  # keep json import used
