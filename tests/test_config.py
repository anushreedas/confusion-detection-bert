import pytest

from confusion_bert.config import Config, apply_overrides, load_config


def test_defaults_match_original_hyperparameters():
    c = Config()
    assert (c.train.epochs, c.train.batch_size, c.train.lr, c.train.warmup_steps) == (
        4,
        16,
        5e-5,
        200,
    )
    assert c.model.classifier_hidden == 50 and c.model.num_classes == 4


def test_overrides_and_roundtrip(tmp_path):
    cfg = load_config(None, ["model.num_layers=6", "train.lr=1e-4", "model.base_model=foo/bar"])
    assert cfg.model.num_layers == 6 and cfg.train.lr == 1e-4 and cfg.model.base_model == "foo/bar"
    cfg.save(tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_unknown_keys_and_bad_values_raise():
    with pytest.raises(ValueError):
        load_config(None, ["model.nope=1"])
    with pytest.raises(ValueError):
        Config.from_dict({"bogus": {}})
    with pytest.raises(ValueError):
        load_config(None, ["model.freeze=sometimes"])
    with pytest.raises(ValueError):
        load_config(None, ["model.freeze=first_n_layers"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["no-equals"])


def test_run_name():
    assert load_config(None, ["model.num_layers=6"]).run_name == "bert-base-uncased_L6"
    c = load_config(
        None, ["model.freeze=first_n_layers", "model.freeze_first_n=6", "model.num_layers=12"]
    )
    assert c.run_name.endswith("_L12_freeze6")
