"""Load a trained run and predict on raw text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from confusion_bert.config import Config, load_config
from confusion_bert.constants import LABEL_NAMES
from confusion_bert.data.dataset import make_collate_fn
from confusion_bert.models.classifier import ConfusionClassifier, build_model_from_run_dir
from confusion_bert.utils import resolve_device


@dataclass(frozen=True)
class Prediction:
    label: int
    label_name: str
    probabilities: list[float]


class ConfusionPredictor:
    def __init__(
        self, model: ConfusionClassifier, tokenizer: object, cfg: Config, device: torch.device
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.cfg = cfg
        self._collate = make_collate_fn(tokenizer, cfg.data.max_length)

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: str = "auto") -> ConfusionPredictor:
        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.yaml")
        model = build_model_from_run_dir(run_dir, cfg.model)
        state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        return cls(model, tokenizer, cfg, resolve_device(device))

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = self._collate([(t, 0) for t in texts[i : i + batch_size]])
            logits = self.model(
                batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
            )
            outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outs:
            return np.zeros((0, self.cfg.model.num_classes))
        return np.concatenate(outs)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[Prediction]:
        probs = self.predict_proba(texts, batch_size)
        return [
            Prediction(
                int(p.argmax()), LABEL_NAMES.get(int(p.argmax()), str(int(p.argmax()))), p.tolist()
            )
            for p in probs
        ]


def default_run_dir() -> str | None:
    return os.environ.get("CONFUSION_BERT_RUN_DIR")
