"""PyTorch dataset and dynamic-padding collate function."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_split(path: str | Path) -> pd.DataFrame:
    """Read a processed split CSV.

    ``keep_default_na=False`` matters: utterances such as "None" or "NA" must not
    be parsed as missing values.
    """
    df = pd.read_csv(path, keep_default_na=False, dtype={"sentence": str})
    missing = {"sentence", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    return df


class SentenceDataset(Dataset[tuple[str, int]]):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]

    @classmethod
    def from_csv(cls, path: str | Path) -> SentenceDataset:
        df = load_split(path)
        return cls(df["sentence"].tolist(), df["label"].tolist())


def make_collate_fn(
    tokenizer: Any, max_length: int
) -> Callable[[list[tuple[str, int]]], dict[str, torch.Tensor]]:
    """Tokenise per batch and pad to the longest sequence in the batch."""

    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch, strict=True)
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out

    return collate
