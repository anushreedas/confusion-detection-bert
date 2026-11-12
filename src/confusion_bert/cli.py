"""Command line interface: ``confusion-bert <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from confusion_bert.config import Config, apply_overrides, load_config
from confusion_bert.utils import setup_logging

log = logging.getLogger("confusion_bert.cli")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/base.yaml", help="YAML config file")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="section.key=value"
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    from confusion_bert.data.labels import load_decoder, load_label_table
    from confusion_bert.data.sentences import build_sentence_table
    from confusion_bert.data.splits import (
        assert_no_text_overlap,
        balance_by_downsampling,
        split_dataset,
    )

    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    out = Path(d.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_sentence_table(
        d.raw_dir,
        load_label_table(d.labels_path),
        load_decoder(d.label_decoder),
        d.terminators,
        d.keep_trailing_fragment,
    )
    df.to_csv(out / "sentences.csv", index=False)
    train, val, test = split_dataset(
        df, d.test_size, d.val_fraction_of_remaining, d.split_strategy, d.seed
    )
    assert_no_text_overlap(train=train, val=val, test=test)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out / f"{name}.csv", index=False)
        log.info(
            "%s: %d rows, class counts %s",
            name,
            len(part),
            part["label"].value_counts().sort_index().to_dict(),
        )
    balance_by_downsampling(train, d.seed).to_csv(out / "train_balanced.csv", index=False)
    balance_by_downsampling(val, d.seed).to_csv(out / "val_balanced.csv", index=False)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from confusion_bert.evaluation.predict import predict_split
    from confusion_bert.training.trainer import split_path, train

    cfg = load_config(args.config, args.overrides)
    run_dir = train(cfg)
    if not args.skip_eval:
        report = predict_split(run_dir, split_path(cfg, "test"), "test")
        log.info("test accuracy=%.4f macro_f1=%.4f", report["accuracy"], report["macro_f1"])
    print(run_dir)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from confusion_bert.config import load_config as lc
    from confusion_bert.evaluation.predict import predict_split
    from confusion_bert.training.trainer import split_path

    cfg = lc(Path(args.run_dir) / "config.yaml")
    csv = Path(args.csv) if args.csv else split_path(cfg, args.split, args.processed_dir)
    report = predict_split(args.run_dir, csv, args.split)
    print(json.dumps({k: report[k] for k in ("n", "accuracy", "macro_f1")}, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from confusion_bert.inference import ConfusionPredictor

    texts = list(args.text or [])
    if args.input:
        texts += [ln.strip() for ln in Path(args.input).read_text().splitlines() if ln.strip()]
    if not texts:
        print("Provide --text or --input", file=sys.stderr)
        return 2
    preds = ConfusionPredictor.from_run_dir(args.run_dir).predict(texts)
    for t, p in zip(texts, preds, strict=True):
        print(
            json.dumps(
                {
                    "text": t,
                    "label": p.label,
                    "label_name": p.label_name,
                    "probabilities": [round(x, 4) for x in p.probabilities],
                }
            )
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np

    from confusion_bert.evaluation.predict import load_predictions
    from confusion_bert.evaluation.significance import pairwise_compare

    cfg = load_config(args.config, args.overrides) if Path(args.config).exists() else Config()
    preds, y_ref = {}, None
    for run in args.runs:
        y, p = load_predictions(run, args.split)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise SystemExit(f"{run}: test labels differ from the first run — different test set?")
        preds[Path(run).name] = p
    assert y_ref is not None
    e = cfg.eval
    table = pairwise_compare(
        y_ref,
        preds,
        e.n_subsamples,
        e.subsample_fraction,
        e.bootstrap_resamples,
        e.alpha,
        e.seed,
        args.metric,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from confusion_bert.evaluation.predict import predict_split
    from confusion_bert.training.trainer import split_path, train

    base = yaml.safe_load(Path(args.config).read_text()) or {}
    experiments = yaml.safe_load(Path(args.experiments).read_text())["experiments"]
    names = args.only or list(experiments)
    for name in names:
        overrides = [f"{k}={v}" for k, v in experiments[name].items()] + [f"train.run_name={name}"]
        cfg = Config.from_dict(apply_overrides(base, overrides + args.overrides))
        log.info("=== experiment %s ===", name)
        run_dir = train(cfg)
        predict_split(run_dir, split_path(cfg, "test"), "test")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="confusion-bert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build sentence dataset and train/val/test splits")
    _add_config_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Fine-tune one configuration, then evaluate on test")
    _add_config_args(p)
    p.add_argument("--skip-eval", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Predict a split with a trained run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--csv", help="Explicit CSV instead of the processed split")
    p.add_argument("--processed-dir")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Classify raw sentences")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--text", action="append")
    p.add_argument("--input", help="File with one sentence per line")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("compare", help="Significance-test trained runs against each other")
    _add_config_args(p)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1"])
    p.add_argument("--out", default="results/comparison.csv")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="Train + evaluate every experiment in a YAML file")
    _add_config_args(p)
    p.add_argument("--experiments", default="configs/experiments.yaml")
    p.add_argument("--only", nargs="*")
    p.set_defaults(func=cmd_sweep)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
