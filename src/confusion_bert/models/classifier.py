"""BERT encoder + small feed-forward head for confusion classification."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, BertModel, PretrainedConfig

from confusion_bert.config import ModelConfig

log = logging.getLogger(__name__)

ENCODER_CONFIG_DIR = "encoder_config"


class ConfusionClassifier(nn.Module):
    """``[CLS]`` hidden state -> Linear(H, 50) -> ReLU -> Linear(50, C).

    The pooler layer is not used (as in the original implementation), so it is
    not instantiated.
    """

    def __init__(
        self,
        encoder: BertModel,
        classifier_hidden: int = 50,
        num_classes: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def build_encoder_config(base_model: str, num_layers: int | None) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(base_model)
    pretrained_depth = config.num_hidden_layers
    if num_layers is not None and num_layers != pretrained_depth:
        if num_layers > pretrained_depth:
            log.warning(
                "num_layers=%d exceeds pretrained depth %d: layers %d-%d are randomly initialised",
                num_layers,
                pretrained_depth,
                pretrained_depth + 1,
                num_layers,
            )
        else:
            log.info("Truncating encoder from %d to %d layers", pretrained_depth, num_layers)
        config.num_hidden_layers = num_layers
    return config


def apply_freeze(model: ConfusionClassifier, mode: str, first_n: int = 0) -> None:
    """Freeze encoder parameters. The classifier head is always trainable."""
    enc = model.encoder
    if mode == "none":
        return
    if mode == "encoder":
        for p in enc.parameters():
            p.requires_grad = False
        return
    if mode == "first_n_layers":
        n_layers = len(enc.encoder.layer)
        if first_n > n_layers:
            raise ValueError(f"freeze_first_n={first_n} exceeds encoder depth {n_layers}")
        for p in enc.embeddings.parameters():
            p.requires_grad = False
        for layer in enc.encoder.layer[:first_n]:
            for p in layer.parameters():
                p.requires_grad = False
        return
    raise ValueError(f"Unknown freeze mode {mode!r}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build_model(cfg: ModelConfig, pretrained: bool = True) -> ConfusionClassifier:
    """Build from a hub name / local path (``pretrained=True``)."""
    if not pretrained:
        raise ValueError("Use build_model_from_run_dir() to build without pretrained weights")
    config = build_encoder_config(cfg.base_model, cfg.num_layers)
    encoder = BertModel.from_pretrained(cfg.base_model, config=config, add_pooling_layer=False)
    model = ConfusionClassifier(encoder, cfg.classifier_hidden, cfg.num_classes, cfg.dropout)
    apply_freeze(model, cfg.freeze, cfg.freeze_first_n)
    trainable, total = count_parameters(model)
    log.info("Model built: %d / %d parameters trainable", trainable, total)
    return model


def build_model_from_run_dir(run_dir: str | Path, cfg: ModelConfig) -> ConfusionClassifier:
    """Rebuild the architecture from the saved encoder config (no downloads)."""
    config = AutoConfig.from_pretrained(Path(run_dir) / ENCODER_CONFIG_DIR)
    encoder = BertModel(config, add_pooling_layer=False)
    return ConfusionClassifier(encoder, cfg.classifier_hidden, cfg.num_classes, cfg.dropout)
