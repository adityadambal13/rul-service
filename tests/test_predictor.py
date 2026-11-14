import copy

import numpy as np
import pandas as pd
import pytest

from rul_service.model import ModelBundle
from rul_service.predictor import InputError, Predictor, health_status
from tests.conftest import readings_from


@pytest.fixture
def predictor(bundle: ModelBundle) -> Predictor:
    return Predictor(bundle)


def test_serving_matches_batch_prediction(
    predictor: Predictor, bundle: ModelBundle, engine_df: pd.DataFrame
) -> None:
    """Training/serving skew check: the API path must equal the batch path."""
    last_cycle = 120
    history = engine_df[engine_df["cycle"] <= last_cycle]  # what batch scoring would see
    batch = bundle.predict_last(history).iloc[0]
    served = predictor.predict("e1", readings_from(engine_df, last_cycle, n=30))
    assert served.rul == pytest.approx(round(float(batch["rul"]), 1))
    assert served.rul_lower == pytest.approx(round(float(batch["rul_lower"]), 1))
    assert served.rul_upper == pytest.approx(round(float(batch["rul_upper"]), 1))
    assert served.cycles_used == 30
    assert served.model_version == bundle.version


def test_full_history_is_accepted_even_if_short(
    predictor: Predictor, engine_df: pd.DataFrame
) -> None:
    result = predictor.predict("e1", readings_from(engine_df, last_cycle=12, n=12))
    assert result.cycles_used == 12


def test_too_little_history_is_rejected(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    with pytest.raises(InputError, match="at least 30"):
        predictor.predict("e1", readings_from(engine_df, last_cycle=100, n=10))


def test_gap_in_cycles_is_rejected(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    readings = readings_from(engine_df, 100)
    del readings[10]
    with pytest.raises(InputError, match="rise by exactly 1"):
        predictor.predict("e1", readings)


def test_missing_field_is_rejected(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    readings = readings_from(engine_df, 100)
    del readings[3]["s_7"]
    with pytest.raises(InputError, match="s_7"):
        predictor.predict("e1", readings)


def test_nan_is_rejected(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    readings = copy.deepcopy(readings_from(engine_df, 100))
    readings[5]["s_3"] = float("nan")
    with pytest.raises(InputError, match="NaN"):
        predictor.predict("e1", readings)


def test_out_of_range_value_is_rejected(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    readings = readings_from(engine_df, 100)
    readings[-1]["s_4"] = 1e6
    with pytest.raises(InputError, match="s_4"):
        predictor.predict("e1", readings)


def test_empty_request_is_rejected(predictor: Predictor) -> None:
    with pytest.raises(InputError, match="empty"):
        predictor.predict("e1", [])


@pytest.mark.parametrize(
    ("rul", "expected"),
    [(5.0, "critical"), (29.9, "critical"), (30.0, "watch"), (59.9, "watch"), (60.0, "healthy")],
)
def test_health_status_bands(rul: float, expected: str) -> None:
    assert health_status(rul) == expected


def test_prediction_is_deterministic(predictor: Predictor, engine_df: pd.DataFrame) -> None:
    a = predictor.predict("e1", readings_from(engine_df, 100))
    b = predictor.predict("e1", readings_from(engine_df, 100))
    assert np.isclose(a.rul, b.rul)
