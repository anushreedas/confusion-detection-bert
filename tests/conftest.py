"""Shared fixtures. Everything runs offline with a tiny randomly-initialised BERT."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from confusion_bert.config import Config, load_config

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = [
    "um",
    "uh",
    "okay",
    "yes",
    "no",
    "what",
    "why",
    "how",
    "i",
    "dont",
    "understand",
    "this",
    "that",
    "makes",
    "sense",
    "right",
    "so",
    "we",
    "can",
    "do",
    "it",
    "huh",
    "confused",
    "clear",
]


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 4-layer, 32-d BERT + tokenizer saved locally, standing in for a hub model."""
    d = tmp_path_factory.mktemp("tiny_bert")
    vocab = {t: i for i, t in enumerate(SPECIAL + WORDS)}
    tk = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    # Built via `tokenizers` because the BertTokenizer(vocab_file=...) API differs across
    # transformers major versions and silently produced an all-[UNK] tokenizer on v5.
    PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(d)
    cfg = BertConfig(
        vocab_size=len(SPECIAL + WORDS),
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(cfg).save_pretrained(d)
    return d


def make_split(n_per_class: int, seed: int) -> pd.DataFrame:
    """Easy synthetic task: the cue word at position 0 determines the class."""
    import random

    rng = random.Random(seed)
    cues = ["okay", "what", "huh", "confused"]
    rows = []
    for label, cue in enumerate(cues):
        for _ in range(n_per_class):
            body = " ".join(rng.choices(WORDS[:20], k=rng.randint(2, 6)))
            rows.append((f"{cue} {body}", label, str(rng.randint(1, 5)), "1"))
    df = pd.DataFrame(rows, columns=["sentence", "label", "participant", "session"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    make_split(40, 1).to_csv(d / "train_balanced.csv", index=False)
    make_split(10, 2).to_csv(d / "val_balanced.csv", index=False)
    make_split(15, 3).to_csv(d / "test.csv", index=False)
    return d


@pytest.fixture
def tiny_cfg(tiny_base: Path, processed_dir: Path, tmp_path: Path) -> Config:
    return load_config(
        None,
        [
            f"model.base_model={tiny_base}",
            "model.num_layers=2",
            f"data.processed_dir={processed_dir}",
            f"train.output_dir={tmp_path / 'runs'}",
            "train.epochs=5",
            "train.batch_size=16",
            "train.lr=3e-3",
            "train.warmup_steps=2",
            "train.device=cpu",
            "data.max_length=32",
        ],
    )


@pytest.fixture
def raw_data(tmp_path: Path) -> dict[str, Path]:
    """Synthetic raw transcripts + label table for the `prepare` command."""
    raw = tmp_path / "raw" / "t1"
    raw.mkdir(parents=True)
    labels = {}
    for pid in ("11", "22", "33", "44"):
        rows = []
        for i in range(160):
            for j in range(3):
                rows.append((f"w{pid}x{i}x{j}", i * 3 + j, i * 3 + j + 1))
            rows.append((".", i * 3 + 2, i * 3 + 3))
        pd.DataFrame(rows, columns=["word", "start", "end"]).to_csv(raw / f"{pid}.csv", index=False)
        labels[f"{pid}t1"] = "1200:1,2400:2,3600:3"
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return {"raw_dir": raw.parent, "labels": path}
