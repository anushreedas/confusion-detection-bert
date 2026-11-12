"""Test helpers (importable as ``tests.helpers``)."""

from __future__ import annotations


def fake_decoder(encoded: str) -> list[tuple[int, int]]:
    """Encoded form for tests: 'time:state,time:state' with times in deciseconds."""
    if not encoded:
        return []
    return [(int(a), int(b)) for a, b in (x.split(":") for x in encoded.split(","))]
