"""FastAPI service. Set ``CONFUSION_BERT_RUN_DIR`` to a trained run directory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from confusion_bert.inference import ConfusionPredictor

MAX_BATCH = 64
MAX_CHARS = 2000


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class PredictionOut(BaseModel):
    label: int
    label_name: str
    probabilities: list[float]


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]


def create_app(run_dir: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        path = run_dir or os.environ.get("CONFUSION_BERT_RUN_DIR")
        if not path:
            raise RuntimeError("Set CONFUSION_BERT_RUN_DIR to a trained run directory")
        app.state.predictor = ConfusionPredictor.from_run_dir(path, device="cpu")
        yield

    app = FastAPI(title="confusion-bert", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if any(len(t) > MAX_CHARS for t in req.texts):
            raise HTTPException(
                status_code=422, detail=f"Each text must be <= {MAX_CHARS} characters"
            )
        preds = app.state.predictor.predict(req.texts)
        return PredictResponse(
            predictions=[
                PredictionOut(label=p.label, label_name=p.label_name, probabilities=p.probabilities)
                for p in preds
            ]
        )

    return app


def app_factory() -> FastAPI:  # pragma: no cover - for `uvicorn --factory`
    return create_app()
