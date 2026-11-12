"""Train/validation/test splitting and class balancing."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _split_once(
    df: pd.DataFrame, test_size: float, strategy: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "stratified":
        counts = df["label"].value_counts()
        if (counts < 2).any():
            raise ValueError(
                f"Cannot stratify: classes with <2 samples: {counts[counts < 2].to_dict()}"
            )
        rest, held = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=seed
        )
    elif strategy == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        rest_idx, held_idx = next(splitter.split(df, groups=df["participant"]))
        rest, held = df.iloc[rest_idx], df.iloc[held_idx]
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")
    return rest.reset_index(drop=True), held.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_fraction_of_remaining: float = 0.16,
    strategy: str = "stratified",
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train, val, test)``.

    ``val_fraction_of_remaining`` is applied *after* removing the test set, so the
    default yields roughly 75.6% / 14.4% / 10% train / val / test.
    """
    train_val, test = _split_once(df, test_size, strategy, seed)
    train, val = _split_once(train_val, val_fraction_of_remaining, strategy, seed)
    return train, val, test


def balance_by_downsampling(df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """Downsample every class to the size of the smallest class, then shuffle."""
    counts = df["label"].value_counts()
    if counts.empty:
        raise ValueError("Cannot balance an empty dataframe")
    n = int(counts.min())
    balanced = df.groupby("label", group_keys=False).sample(n=n, random_state=seed)
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assert_no_text_overlap(**splits: pd.DataFrame) -> None:
    """Raise if any sentence text appears in more than one split (leakage check)."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]["sentence"]) & set(splits[b]["sentence"])
            if overlap:
                raise AssertionError(f"{len(overlap)} sentences leak between '{a}' and '{b}'")
