from pathlib import Path

import pandas as pd

from rul_service.model import ModelBundle, evaluate_bundle, load_bundle, save_bundle


def test_intervals_surround_the_point_estimate(bundle: ModelBundle, train_df: pd.DataFrame) -> None:
    pred = bundle.predict_last(train_df)
    assert (pred["rul_lower"] <= pred["rul"]).all()
    assert (pred["rul"] <= pred["rul_upper"]).all()
    assert pred["rul"].between(0, bundle.rul_cap).all()
    assert len(pred) == train_df["unit"].nunique()


def test_save_and_load_give_identical_predictions(
    bundle: ModelBundle, train_df: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "b.joblib"
    save_bundle(bundle, path)
    reloaded = load_bundle(path)
    pd.testing.assert_frame_equal(bundle.predict_last(train_df), reloaded.predict_last(train_df))
    assert reloaded.version == bundle.version


def test_constant_sensors_are_dropped(bundle: ModelBundle) -> None:
    assert "s_1" not in bundle.scaler.kept_sensors
    assert "s_5" not in bundle.scaler.kept_sensors
    assert len(bundle.scaler.kept_sensors) == 19


def test_evaluate_bundle_returns_sane_metrics(bundle: ModelBundle, train_df: pd.DataFrame) -> None:
    # Cut every engine 30 cycles before its end; the true RUL is then exactly 30.
    last = train_df.groupby("unit")["cycle"].transform("max")
    cut = train_df[train_df["cycle"] <= last - 30]
    truth = pd.DataFrame({"unit": sorted(train_df["unit"].unique()), "rul": 30})
    metrics = evaluate_bundle(bundle, cut, truth)
    assert set(metrics) == {
        "test_rmse",
        "test_nasa_score",
        "interval_coverage",
        "mean_interval_width",
    }
    assert 0.0 <= metrics["interval_coverage"] <= 1.0
    assert metrics["test_rmse"] >= 0.0
