import pandas as pd
import pytest
import yaml

from confusion_bert.cli import main


def _cfg(tmp_path, raw_data, strategy="stratified"):
    cfg = {
        "data": {
            "raw_dir": str(raw_data["raw_dir"]),
            "labels_path": str(raw_data["labels"]),
            "processed_dir": str(tmp_path / "proc"),
            "label_decoder": "tests.helpers:fake_decoder",
            "split_strategy": strategy,
        }
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_prepare_end_to_end(tmp_path, raw_data):
    assert main(["prepare", "--config", str(_cfg(tmp_path, raw_data))]) == 0
    proc = tmp_path / "proc"
    allx = pd.read_csv(proc / "sentences.csv")
    assert len(allx) == 640 and sorted(allx.label.unique()) == [0, 1, 2, 3]
    tr, va, te = (pd.read_csv(proc / f"{n}.csv") for n in ("train", "val", "test"))
    assert len(tr) + len(va) + len(te) == 640
    assert len(te) == 64
    trb = pd.read_csv(proc / "train_balanced.csv")
    assert trb.label.value_counts().nunique() == 1


def test_prepare_group_strategy_holds_out_participants(tmp_path, raw_data):
    assert (
        main(
            [
                "prepare",
                "--config",
                str(_cfg(tmp_path, raw_data, "group")),
                "--set",
                "data.test_size=0.25",
            ]
        )
        == 0
    )
    proc = tmp_path / "proc"
    tr = pd.read_csv(proc / "train.csv", dtype=str)
    te = pd.read_csv(proc / "test.csv", dtype=str)
    assert not set(tr.participant) & set(te.participant)


def test_missing_course_decoder_gives_actionable_error(tmp_path, raw_data):
    cfg = _cfg(tmp_path, raw_data)
    raw = yaml.safe_load(cfg.read_text())
    raw["data"]["label_decoder"] = "confusion_bert.data.labels:course_base85_decoder"
    cfg.write_text(yaml.safe_dump(raw))
    from confusion_bert.data.labels import LabelDecoderUnavailable

    with pytest.raises(LabelDecoderUnavailable, match="label_decoder"):
        main(["prepare", "--config", str(cfg)])
