import logging

import pytest
import torch

from confusion_bert.config import ModelConfig
from confusion_bert.models.classifier import apply_freeze, build_model, count_parameters


def _model(base, **kw):
    return build_model(ModelConfig(base_model=str(base), **kw))


def test_truncation_and_forward_shape(tiny_base):
    m = _model(tiny_base, num_layers=2)
    assert len(m.encoder.encoder.layer) == 2
    ids = torch.randint(5, 20, (3, 7))
    out = m(ids, torch.ones_like(ids))
    assert out.shape == (3, 4)


def test_extension_adds_random_layers_and_warns(tiny_base, caplog):
    with caplog.at_level(logging.WARNING):
        m = _model(tiny_base, num_layers=6)
    assert len(m.encoder.encoder.layer) == 6
    assert "randomly initialised" in caplog.text


def test_default_depth_and_no_pooler(tiny_base):
    m = _model(tiny_base)
    assert len(m.encoder.encoder.layer) == 4
    assert getattr(m.encoder, "pooler", None) is None


def test_pretrained_weights_are_loaded(tiny_base):
    from transformers import BertModel

    ref = BertModel.from_pretrained(tiny_base)
    m = _model(tiny_base)
    assert torch.equal(
        ref.embeddings.word_embeddings.weight, m.encoder.embeddings.word_embeddings.weight
    )


def test_freeze_encoder(tiny_base):
    m = _model(tiny_base, freeze="encoder")
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for p in m.classifier.parameters())
    trainable, total = count_parameters(m)
    assert 0 < trainable < total


def test_freeze_first_n_layers(tiny_base):
    m = _model(tiny_base, freeze="first_n_layers", freeze_first_n=2)
    layers = m.encoder.encoder.layer
    assert not any(p.requires_grad for p in m.encoder.embeddings.parameters())
    assert not any(p.requires_grad for L in layers[:2] for p in L.parameters())
    assert all(p.requires_grad for L in layers[2:] for p in L.parameters())


def test_freeze_first_n_too_large(tiny_base):
    m = _model(tiny_base)
    with pytest.raises(ValueError):
        apply_freeze(m, "first_n_layers", 99)


def test_fixture_tokenizer_is_not_all_unk(tiny_base):
    """Guard against a vacuous fixture: known words must map to real, distinct ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_base)
    ids = tok("okay what huh confused")["input_ids"]
    assert tok.unk_token_id not in ids and len(set(ids)) == 6  # CLS + 4 words + SEP
