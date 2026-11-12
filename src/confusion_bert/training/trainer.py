"""Fine-tuning loop with per-epoch validation and best-epoch checkpointing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from confusion_bert.config import Config
from confusion_bert.data.dataset import SentenceDataset, make_collate_fn
from confusion_bert.evaluation.metrics import accuracy, macro_f1
from confusion_bert.models.classifier import (
    ENCODER_CONFIG_DIR,
    ConfusionClassifier,
    build_model,
    count_parameters,
)
from confusion_bert.utils import resolve_device, seed_everything

log = logging.getLogger(__name__)


def split_path(cfg: Config, split: str, processed_dir: str | Path | None = None) -> Path:
    """Path of a processed split, honouring the balancing flags for train/val."""
    root = Path(processed_dir or cfg.data.processed_dir)
    balanced = {"train": cfg.data.balance_train, "val": cfg.data.balance_val}.get(split, False)
    return root / f"{split}_balanced.csv" if balanced else root / f"{split}.csv"


@torch.no_grad()
def _evaluate(
    model: ConfusionClassifier, loader: DataLoader[Any], loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss, ys, ps = 0.0, [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        labels = batch["labels"].to(device)
        total_loss += loss_fn(logits, labels).item()
        ys.append(labels.cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "val_accuracy": accuracy(y, p),
        "val_macro_f1": macro_f1(y, p),
    }


def train(
    cfg: Config, processed_dir: str | Path | None = None, run_dir: str | Path | None = None
) -> Path:
    """Fine-tune one model configuration and return the run directory."""
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    run_dir = Path(run_dir or Path(cfg.train.output_dir) / cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    log.info("Run directory: %s | device: %s", run_dir, device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    collate = make_collate_fn(tokenizer, cfg.data.max_length)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    train_ds = SentenceDataset.from_csv(split_path(cfg, "train", processed_dir))
    val_ds = SentenceDataset.from_csv(split_path(cfg, "val", processed_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
    )

    model = build_model(cfg.model).to(device)
    trainable, total = count_parameters(model)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        eps=cfg.train.eps,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = len(train_loader) * cfg.train.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.train.warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_f1, best_epoch = -1.0, -1
    start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        row = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1)}
        row.update(_evaluate(model, val_loader, loss_fn, device))
        history.append(row)
        log.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            cfg.train.epochs,
            row["train_loss"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_macro_f1"],
        )
        improved = row["val_macro_f1"] > best_f1
        if cfg.train.select == "last" or improved:
            best_f1, best_epoch = row["val_macro_f1"], epoch
            torch.save(model.state_dict(), run_dir / "model.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    model.encoder.config.save_pretrained(run_dir / ENCODER_CONFIG_DIR)
    summary = {
        "selected_epoch": best_epoch,
        "selection": cfg.train.select,
        "selected_val": next(h for h in history if h["epoch"] == best_epoch),
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(time.time() - start, 1),
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
