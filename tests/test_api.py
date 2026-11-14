from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from tests.conftest import readings_from


def _body(engine_df: pd.DataFrame, last_cycle: int = 100) -> dict[str, Any]:
    return {"engine_id": "engine-7", "readings": readings_from(engine_df, last_cycle)}


def test_health_is_always_ok(client_without_model: TestClient) -> None:
    assert client_without_model.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_not_ready_and_503_without_model(
    client_without_model: TestClient, engine_df: pd.DataFrame
) -> None:
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/v1/predict", json=_body(engine_df)).status_code == 503


def test_predict_happy_path(client: TestClient, engine_df: pd.DataFrame) -> None:
    r = client.post("/v1/predict", json=_body(engine_df))
    assert r.status_code == 200
    data = r.json()
    assert data["engine_id"] == "engine-7"
    assert data["rul_lower"] <= data["rul"] <= data["rul_upper"]
    assert data["status"] in {"healthy", "watch", "critical"}
    assert data["cycles_used"] == 30
    assert r.headers["X-Request-ID"]


def test_request_id_is_echoed(client: TestClient) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_missing_field_is_422(client: TestClient, engine_df: pd.DataFrame) -> None:
    body = _body(engine_df)
    del body["readings"][0]["s_2"]
    assert client.post("/v1/predict", json=body).status_code == 422


def test_unknown_field_is_422(client: TestClient, engine_df: pd.DataFrame) -> None:
    body = _body(engine_df)
    body["readings"][0]["surprise"] = 1.0
    assert client.post("/v1/predict", json=body).status_code == 422


def test_out_of_range_is_422_with_reason(client: TestClient, engine_df: pd.DataFrame) -> None:
    body = _body(engine_df)
    body["readings"][-1]["s_4"] = 1e6
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 422
    assert "s_4" in r.json()["detail"]


def test_short_history_is_422(client: TestClient, engine_df: pd.DataFrame) -> None:
    body = {"engine_id": "e", "readings": readings_from(engine_df, 100, n=5)}
    assert client.post("/v1/predict", json=body).status_code == 422


def test_model_info(client: TestClient) -> None:
    r = client.get("/v1/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["subset"] == "FD001"
    assert info["window"] == 30
