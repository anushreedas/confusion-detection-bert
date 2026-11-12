import json

import numpy as np
import pandas as pd
import torch
from transformers import BertModel

from confusion_bert.cli import main
from confusion_bert.config import load_config
from confusion_bert.evaluation.predict import load_predictions, predict_split
from confusion_bert.inference import ConfusionPredictor
from confusion_bert.training.trainer import split_path, train


def test_train_writes_artifacts_and_learns(tiny_cfg):
    run = train(tiny_cfg)
    for name in [
        "model.pt",
        "config.yaml",
        "history.csv",
        "train_summary.json",
        "tokenizer",
        "encoder_config",
    ]:
        assert (run / name).exists(), name
    hist = pd.read_csv(run / "history.csv")
    assert len(hist) == tiny_cfg.train.epochs
    assert hist.train_loss.iloc[-1] < hist.train_loss.iloc[0]
    summary = json.loads((run / "train_summary.json").read_text())
    assert summary["selected_val"]["val_macro_f1"] == hist.val_macro_f1.max()
    assert summary["trainable_params"] == summary["total_params"]


def test_predictor_roundtrip_and_split_prediction(tiny_cfg):
    run = train(tiny_cfg)
    report = predict_split(run, split_path(tiny_cfg, "test"), "test")
    assert report["n"] == 60 and report["accuracy"] > 0.5  # easy synthetic cue task

    pred = ConfusionPredictor.from_run_dir(run, device="cpu")
    texts = pd.read_csv(split_path(tiny_cfg, "test"), keep_default_na=False)["sentence"].tolist()
    probs = pred.predict_proba(texts)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    _, y_pred = load_predictions(run, "test")
    np.testing.assert_array_equal(probs.argmax(axis=1), y_pred)
    one = pred.predict(["okay yes"])[0]
    assert one.label in range(4) and len(one.probabilities) == 4


def test_frozen_encoder_weights_do_not_change(tiny_cfg, tiny_base):
    cfg = load_config(
        None, [f"model.base_model={tiny_base}", "model.num_layers=2", "model.freeze=encoder"]
    )
    cfg.data, cfg.train = tiny_cfg.data, tiny_cfg.train
    run = train(cfg)
    ref = BertModel.from_pretrained(tiny_base).embeddings.word_embeddings.weight
    saved = torch.load(run / "model.pt", weights_only=True)[
        "encoder.embeddings.word_embeddings.weight"
    ]
    assert torch.equal(ref, saved)
    assert json.loads((run / "train_summary.json").read_text())["trainable_params"] < 10_000


def test_cli_train_compare_roundtrip(tiny_cfg, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    tiny_cfg.save(cfg_path)
    runs = []
    for layers in (1, 2):
        rc = main(
            [
                "train",
                "--config",
                str(cfg_path),
                "--set",
                f"model.num_layers={layers}",
                "--set",
                "train.epochs=2",
            ]
        )
        assert rc == 0
        runs.append(capsys.readouterr().out.strip().splitlines()[-1])
    out = tmp_path / "cmp.csv"
    assert (
        main(
            [
                "compare",
                "--config",
                str(cfg_path),
                "--runs",
                *runs,
                "--out",
                str(out),
                "--set",
                "eval.bootstrap_resamples=50",
                "--set",
                "eval.n_subsamples=5",
            ]
        )
        == 0
    )
    table = pd.read_csv(out)
    assert len(table) == 2 and {"sign_p", "mcnemar_p", "boot_ci_low", "sign_p_holm"} <= set(
        table.columns
    )
    assert main(["predict", "--run-dir", runs[0], "--text", "okay yes"]) == 0
    assert '"label_name"' in capsys.readouterr().out


def test_cli_sweep_and_evaluate(tiny_cfg, tmp_path, capsys):
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    tiny_cfg.save(cfg_path)
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        yaml.safe_dump(
            {
                "experiments": {
                    "L1": {"model.num_layers": 1},
                    "L2_freeze": {"model.num_layers": 2, "model.freeze": "encoder"},
                }
            }
        )
    )
    rc = main(
        ["sweep", "--config", str(cfg_path), "--experiments", str(exp), "--set", "train.epochs=1"]
    )
    assert rc == 0
    runs = tmp_path / "runs"
    assert {p.name for p in runs.iterdir()} == {"L1", "L2_freeze"}
    for name in ("L1", "L2_freeze"):
        assert (runs / name / "metrics_test.json").exists()

    capsys.readouterr()
    assert (
        main(
            [
                "sweep",
                "--config",
                str(cfg_path),
                "--experiments",
                str(exp),
                "--only",
                "L1",
                "--set",
                "train.epochs=1",
            ]
        )
        == 0
    )
    assert main(["evaluate", "--run-dir", str(runs / "L1")]) == 0
    assert json.loads(capsys.readouterr().out.split("\n}\n")[-2] + "\n}")["n"] == 60
