from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from tests.conftest import readings_from


def _body(engine_df: pd.DataFrame) -> dict[str, Any]:
    return {"engine_id": "engine-7", "readings": readings_from(engine_df, 100)}


def test_predict_without_key_is_401_when_key_required(
    keyed_client: TestClient, engine_df: pd.DataFrame
) -> None:
    assert keyed_client.post("/v1/predict", json=_body(engine_df)).status_code == 401


def test_predict_with_wrong_key_is_401(keyed_client: TestClient, engine_df: pd.DataFrame) -> None:
    r = keyed_client.post("/v1/predict", json=_body(engine_df), headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_predict_with_right_key_works(keyed_client: TestClient, engine_df: pd.DataFrame) -> None:
    r = keyed_client.post("/v1/predict", json=_body(engine_df), headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_model_info_is_protected_too(keyed_client: TestClient) -> None:
    assert keyed_client.get("/v1/model-info").status_code == 401
    assert keyed_client.get("/v1/model-info", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_health_and_ready_stay_open(keyed_client: TestClient) -> None:
    assert keyed_client.get("/health").status_code == 200
    assert keyed_client.get("/ready").status_code == 200


def test_no_key_configured_means_open(client: TestClient, engine_df: pd.DataFrame) -> None:
    assert client.post("/v1/predict", json=_body(engine_df)).status_code == 200
