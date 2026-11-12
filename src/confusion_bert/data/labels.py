"""Decode confusion annotations and align them to word-level timestamps.

The study stores each participant/session's confusion annotation as an encoded
string (``data/labels/conf_labels.json``). Decoding yields *change points*: pairs
``(decisecond, state)`` meaning "from this time on the participant reported
``state``". Each transcript token then takes the state of the latest change point
at or before its timestamp.

Decoding the strings needs the course-provided ``base85`` helper module
(``time16_decode``), which is not distributed with this repository. The decoder
is therefore a plug-in: set ``data.label_decoder`` to any ``module:callable``.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

ChangePoint = tuple[int, int]
Decoder = Callable[[str], Sequence[ChangePoint]]


class LabelDecoderUnavailable(RuntimeError):
    """Raised when the label decoder cannot be imported."""


def course_base85_decoder(encoded: str) -> list[ChangePoint]:
    """Decode with the course-provided ``base85.time16_decode`` helper.

    Mirrors the original notebook: the first element of the decoder output is
    discarded, the remainder are ``(decisecond, state)`` change points.
    """
    try:
        import base85
    except ImportError as exc:  # pragma: no cover - depends on external module
        raise LabelDecoderUnavailable(
            "The `base85` module providing `time16_decode` is not installed. It was "
            "supplied with the original course materials and is not on PyPI. Install it, "
            "or point `data.label_decoder` at your own `module:callable` returning "
            "[(decisecond, state), ...]."
        ) from exc
    return list(base85.time16_decode(encoded)[1:])


def load_decoder(spec: str) -> Decoder:
    """Resolve a ``module:callable`` string."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Decoder spec must look like 'module:callable', got {spec!r}")
    module = importlib.import_module(module_name)
    decoder = getattr(module, attr)
    if not callable(decoder):
        raise TypeError(f"{spec} is not callable")
    return decoder


def load_label_table(path: str | Path) -> dict[str, str]:
    """Load ``{"<participant>t<session>": "<encoded labels>"}``."""
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    if not isinstance(table, dict) or not all(isinstance(v, str) for v in table.values()):
        raise ValueError(f"{path} must contain a JSON object of string values")
    return table


def label_key(csv_path: str | Path) -> str:
    """Key into the label table for a transcript at ``.../t<session>/<participant>.csv``."""
    p = Path(csv_path)
    return f"{p.stem}t{p.parent.name[-1]}"


def participant_and_session(csv_path: str | Path) -> tuple[str, str]:
    p = Path(csv_path)
    return p.stem, p.parent.name[-1]


def to_seconds(change_points: Iterable[ChangePoint]) -> list[ChangePoint]:
    """Convert deciseconds to whole seconds (ceil), as in the original pipeline."""
    return [(math.ceil(t / 10), int(v)) for t, v in change_points]


def token_labels(timestamps: ArrayLike, change_points_sec: Iterable[ChangePoint]) -> np.ndarray:
    """Per-token confusion state.

    A token at time ``t`` gets the state of the latest change point with start time
    ``<= t``. A change point ``(0, 0)`` is always present, so tokens before the
    first annotation are "not confused". When two change points share a timestamp
    the larger state wins.
    """
    merged: dict[int, int] = {0: 0}
    for t, s in change_points_sec:
        merged[t] = max(merged.get(t, s), s)
    times = np.array(sorted(merged), dtype=np.int64)
    states = np.array([merged[int(t)] for t in times], dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.float64)
    idx = np.searchsorted(times, ts, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return states[idx]
