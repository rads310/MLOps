"""HTTP-интерфейс к модели: минимальный сервис на FastAPI."""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.featurize import build
from src.predict import MODEL_FILE
from src.utils import load_params

app = FastAPI(title="Bike Sharing — почасовой прогноз аренд", version="1.0")


class Observation(BaseModel):
    hr: int = Field(ge=0, le=23)
    season: int = Field(ge=1, le=4)
    yr: int = Field(ge=0, le=1)
    mnth: int = Field(ge=1, le=12)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0.0, le=1.0)
    atemp: float = Field(ge=0.0, le=1.0)
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    records: list[Observation]


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str


@lru_cache(maxsize=1)
def get_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"], bundle["features"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    params = load_params()
    model, features = get_model()

    frame = pd.DataFrame([r.model_dump() for r in request.records])
    built = build(
        frame,
        target=params["data"]["target"],
        drop=list(params["data"]["drop"]),
        cyclic_hour=bool(params["featurize"]["cyclic_hour"]),
    )
    x_new = built.reindex(columns=features)
    predictions = model.predict(x_new)
    return PredictResponse(
        predictions=[round(float(p), 1) for p in predictions],
        model_version=str(MODEL_FILE),
    )
