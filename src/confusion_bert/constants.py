"""Shared constants."""

from __future__ import annotations

#: Confusion levels as defined by the study participants' self-annotation.
LABEL_NAMES: dict[int, str] = {
    0: "not_confused",
    1: "slightly_confused",
    2: "very_confused",
    3: "extremely_confused",
}
NUM_CLASSES = len(LABEL_NAMES)

#: Column order of the processed sentence CSVs.
SENTENCE_COLUMNS = ["sentence", "label", "participant", "session"]
