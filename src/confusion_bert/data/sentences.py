"""Turn timestamped word tokens into labelled, de-duplicated sentences."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from confusion_bert.constants import SENTENCE_COLUMNS
from confusion_bert.data.labels import (
    Decoder,
    label_key,
    participant_and_session,
    to_seconds,
    token_labels,
)

log = logging.getLogger(__name__)


def read_transcript(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read ``(word, start_time_seconds, ...)`` rows from a transcript CSV.

    Expects a header row and the word and start time in the first two columns
    (further columns are ignored). Missing words become ``""`` with time 0;
    missing times become 0 — matching the original preprocessing.
    """
    df = pd.read_csv(path, header=0, usecols=[0, 1, 2])
    words = df.iloc[:, 0]
    times = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    missing_word = words.isna()
    times = times.mask(missing_word, 0).fillna(0)
    return words.fillna("").astype(str).tolist(), times.to_numpy(dtype=np.float64)


def segment(
    words: Sequence[str],
    labels: Iterable[int],
    terminators: Iterable[str] = (".", "?"),
    keep_trailing_fragment: bool = False,
) -> list[tuple[str, int]]:
    """Split tokens into sentences on terminator tokens.

    A sentence's label is the **maximum** token label inside it ("as confused as
    its most confused moment"). Empty sentences (consecutive terminators) are
    skipped. Tokens after the last terminator are dropped unless
    ``keep_trailing_fragment`` is set.
    """
    term = set(terminators)
    out: list[tuple[str, int]] = []
    cur_words: list[str] = []
    cur_labels: list[int] = []
    for word, label in zip(words, labels, strict=True):
        if word in term:
            if cur_words:
                out.append((" ".join(cur_words), max(cur_labels)))
            cur_words, cur_labels = [], []
        elif word != "":
            cur_words.append(word)
            cur_labels.append(int(label))
    if keep_trailing_fragment and cur_words:
        out.append((" ".join(cur_words), max(cur_labels)))
    return out


def build_sentence_table(
    raw_dir: str | Path,
    label_table: dict[str, str],
    decoder: Decoder,
    terminators: Iterable[str] = (".", "?"),
    keep_trailing_fragment: bool = False,
) -> pd.DataFrame:
    """Build the global, de-duplicated sentence table from all transcripts.

    Files are processed in sorted order so the result is deterministic. A sentence
    already seen in any earlier transcript is dropped (first occurrence wins),
    which prevents identical utterances ending up in both train and test.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No transcript CSVs found under {raw_dir}")

    seen: set[str] = set()
    rows: list[tuple[str, int, str, str]] = []
    n_dupes = 0
    for path in files:
        key = label_key(path)
        if key not in label_table:
            raise KeyError(f"No annotation for {path} (expected key {key!r} in label table)")
        words, times = read_transcript(path)
        change_points = to_seconds(decoder(label_table[key]))
        tok = token_labels(times, change_points)
        participant, session = participant_and_session(path)
        for text, label in segment(words, tok, terminators, keep_trailing_fragment):
            if text in seen:
                n_dupes += 1
                continue
            seen.add(text)
            rows.append((text, label, participant, session))

    df = pd.DataFrame(rows, columns=SENTENCE_COLUMNS)
    log.info(
        "Built %d sentences from %d files (%d duplicates dropped)", len(df), len(files), n_dupes
    )
    return df
