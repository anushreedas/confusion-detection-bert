import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from confusion_bert.serving.app import create_app
from confusion_bert.training.trainer import train


@pytest.fixture
def client(tiny_cfg):
    tiny_cfg.train.epochs = 1
    run = train(tiny_cfg)
    with TestClient(create_app(str(run))) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json={"texts": ["okay yes", "what huh"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert abs(sum(preds[0]["probabilities"]) - 1) < 1e-5


@pytest.mark.parametrize(
    "payload", [{"texts": []}, {"texts": ["x" * 2001]}, {"texts": ["a"] * 65}, {}]
)
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
