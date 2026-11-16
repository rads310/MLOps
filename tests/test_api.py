from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import MODEL_FILE, app

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="модель не обучена")
client = TestClient(app)

RECORD = {
    "hr": 8,
    "season": 1,
    "yr": 0,
    "mnth": 1,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.24,
    "atemp": 0.2879,
    "hum": 0.81,
    "windspeed": 0.0,
}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_one_number_per_record() -> None:
    response = client.post("/predict", json={"records": [RECORD, RECORD]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(p >= 0 for p in predictions)


def test_invalid_hour_is_rejected() -> None:
    bad = dict(RECORD, hr=42)
    response = client.post("/predict", json={"records": [bad]})
    assert response.status_code == 422
