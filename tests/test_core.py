import numpy as np
import pandas as pd
import pytest

from rul_service.data import COLUMNS, SENSOR_COLS, SETTING_COLS
from rul_service.features import RegimeScaler, build_features, last_rows
from rul_service.labels import add_rul
from rul_service.metrics import nasa_score, rmse


def _toy(n_units: int = 2, n_cycles: int = 6) -> pd.DataFrame:
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            rows.append([u, c, 0.0, 0.0, 100.0] + [float(c * (k + 1)) for k in range(21)])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_add_rul_counts_down_to_zero() -> None:
    out = add_rul(_toy(), cap=None)
    for _, g in out.groupby("unit"):
        assert g["rul"].iloc[-1] == 0
        assert g["rul"].iloc[0] == len(g) - 1


def test_add_rul_cap_applied() -> None:
    out = add_rul(_toy(n_cycles=10), cap=3)
    assert out["rul"].max() == 3


def test_metrics_perfect_prediction() -> None:
    y = np.array([10.0, 50.0, 90.0])
    assert rmse(y, y) == 0.0
    assert nasa_score(y, y) == 0.0


def test_nasa_score_penalises_late_more_than_early() -> None:
    assert nasa_score([50.0], [60.0]) > nasa_score([50.0], [40.0])


def test_build_features_uses_only_past_and_has_no_nans() -> None:
    df = _toy(n_cycles=8)
    feats = build_features(df, ["s_1"], window=3)
    assert not feats.isna().any().any()
    unit1 = feats[df["unit"] == 1]
    # s_1 at cycle c equals c, so rolling mean over 3 cycles at cycle 5 is (3+4+5)/3
    assert unit1["s_1_mean"].iloc[4] == pytest.approx(4.0)
    # delta at cycle 5 is value now minus value 2 cycles ago
    assert unit1["s_1_delta"].iloc[4] == pytest.approx(2.0)
    # early rows fall back to the engine's first reading
    assert unit1["s_1_delta"].iloc[0] == 0.0


def test_last_rows_selects_one_row_per_engine() -> None:
    df = _toy(n_units=3)
    assert last_rows(df).sum() == 3


def test_regime_scaler_separates_regimes() -> None:
    rng = np.random.default_rng(0)
    parts = []
    for regime_offset, setting in [(0.0, 0.0), (500.0, 10.0)]:
        d = _toy(n_units=3, n_cycles=20)
        d[SETTING_COLS[0]] = setting
        d[SENSOR_COLS] = d[SENSOR_COLS] + regime_offset + rng.normal(0, 1, d[SENSOR_COLS].shape)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    scaler = RegimeScaler(n_regimes=2).fit(df)
    scaled = scaler.transform(df)
    # after per-regime scaling, the 500-unit regime offset must be gone
    assert abs(scaled["s_5"].mean()) < 0.1
    assert scaled["s_5"].abs().max() < 10
