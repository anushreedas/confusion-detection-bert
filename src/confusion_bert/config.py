"""Typed, validated experiment configuration.

Configs are plain YAML files that map onto the dataclasses below. Any value can
be overridden from the CLI with ``--set section.key=value``; unknown keys raise
instead of being silently ignored.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

FREEZE_MODES = ("none", "encoder", "first_n_layers")
SPLIT_STRATEGIES = ("stratified", "group")
SELECT_MODES = ("best_val_macro_f1", "last")


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    labels_path: str = "data/labels/conf_labels.json"
    processed_dir: str = "data/processed"
    #: ``module:callable`` returning ``[(decisecond, state), ...]`` for an encoded label string.
    label_decoder: str = "confusion_bert.data.labels:course_base85_decoder"
    terminators: list[str] = field(default_factory=lambda: [".", "?"])
    keep_trailing_fragment: bool = False
    test_size: float = 0.10
    #: Fraction of the non-test data held out for validation.
    val_fraction_of_remaining: float = 0.16
    #: ``stratified`` = random sentence-level split (original protocol);
    #: ``group`` = hold out whole participants (stricter, recommended).
    split_strategy: str = "stratified"
    balance_train: bool = True
    balance_val: bool = True
    max_length: int = 512
    seed: int = 123

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("data.test_size must be in (0, 1)")
        if not 0 < self.val_fraction_of_remaining < 1:
            raise ValueError("data.val_fraction_of_remaining must be in (0, 1)")
        if self.split_strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"data.split_strategy must be one of {SPLIT_STRATEGIES}")
        if self.max_length < 8:
            raise ValueError("data.max_length must be >= 8")


@dataclass
class ModelConfig:
    base_model: str = "bert-base-uncased"
    #: Number of encoder layers. ``None`` keeps the pretrained depth. Fewer layers
    #: truncates the stack; more layers appends randomly-initialised layers.
    num_layers: int | None = None
    freeze: str = "none"
    #: Used when ``freeze == "first_n_layers"``: freeze embeddings + the first N layers.
    freeze_first_n: int = 0
    classifier_hidden: int = 50
    num_classes: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.freeze not in FREEZE_MODES:
            raise ValueError(f"model.freeze must be one of {FREEZE_MODES}")
        if self.num_layers is not None and self.num_layers < 1:
            raise ValueError("model.num_layers must be >= 1")
        if self.freeze == "first_n_layers" and self.freeze_first_n < 1:
            raise ValueError("model.freeze_first_n must be >= 1 when freeze='first_n_layers'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float = 5e-5
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    #: ``best_val_macro_f1`` keeps the best epoch; ``last`` keeps the final epoch
    #: (what the original notebooks did).
    select: str = "best_val_macro_f1"
    output_dir: str = "runs"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("train.epochs and train.batch_size must be >= 1")
        if self.select not in SELECT_MODES:
            raise ValueError(f"train.select must be one of {SELECT_MODES}")


@dataclass
class EvalConfig:
    n_subsamples: int = 20
    subsample_fraction: float = 0.3
    seed: int = 0
    bootstrap_resamples: int = 10_000
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.subsample_fraction < 1:
            raise ValueError("eval.subsample_fraction must be in (0, 1)")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        raw = copy.deepcopy(raw or {})
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        return cls(
            data=_build(DataConfig, raw.get("data")),
            model=_build(ModelConfig, raw.get("model")),
            train=_build(TrainConfig, raw.get("train")),
            eval=_build(EvalConfig, raw.get("eval")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @property
    def run_name(self) -> str:
        if self.train.run_name:
            return self.train.run_name
        base = self.model.base_model.rstrip("/").split("/")[-1]
        layers = f"L{self.model.num_layers}" if self.model.num_layers else "Ldefault"
        freeze = {
            "none": "",
            "encoder": "_freezeall",
            "first_n_layers": f"_freeze{self.model.freeze_first_n}",
        }[self.model.freeze]
        return f"{base}_{layers}{freeze}"


def _build(cls: type[T], raw: dict[str, Any] | None) -> T:
    raw = raw or {}
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        # PyYAML follows YAML 1.1 and reads "1e-4" as a string; coerce to the declared type.
        if isinstance(value, str) and hints[key] in (float, int):
            try:
                raw[key] = hints[key](value)
            except ValueError as exc:
                raise ValueError(f"{cls.__name__}.{key}: cannot parse {value!r}") from exc
    return cls(**raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML scalars)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.key, got {dotted!r}")
        section, key = parts
        out.setdefault(section, {})
        out[section][key] = yaml.safe_load(value)
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.from_dict(apply_overrides(raw, overrides or []))
