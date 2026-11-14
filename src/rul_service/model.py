"""Model bundle: everything needed to turn raw sensor readings into a RUL prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rul_service import config
from rul_service.data import SENSOR_COLS, SETTING_COLS
from rul_service.features import RegimeScaler, build_features, last_rows
from rul_service.labels import add_rul
from rul_service.metrics import nasa_score, rmse


@dataclass
class ModelBundle:
    """Scaler + three models + the settings that must match between training and serving."""

    scaler: RegimeScaler
    point_model: HistGradientBoostingRegressor
    lower_model: HistGradientBoostingRegressor
    upper_model: HistGradientBoostingRegressor
    feature_cols: list[str]
    window: int
    rul_cap: int
    input_ranges: dict[str, tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict RUL at the final recorded cycle of every engine in `df`.

        Returns a frame indexed by unit with columns rul, rul_lower, rul_upper.
        """
        scaled = self.scaler.transform(df)
        feats = build_features(scaled, self.scaler.kept_sensors, self.window)
        mask = last_rows(df)
        x = feats.loc[mask, self.feature_cols]
        units = df.loc[mask, "unit"].to_numpy()

        point = np.clip(self.point_model.predict(x), 0, self.rul_cap)
        lower = np.clip(self.lower_model.predict(x), 0, self.rul_cap)
        upper = np.clip(self.upper_model.predict(x), 0, self.rul_cap)
        lower = np.minimum(lower, point)  # keep the band around the point estimate
        upper = np.maximum(upper, point)

        out = pd.DataFrame(
            {"rul": point, "rul_lower": lower, "rul_upper": upper},
            index=pd.Index(units, name="unit"),
        )
        return out.sort_index()


def _regressor(max_iter: int, **kwargs: Any) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.05, random_state=0, **kwargs
    )


def fit_bundle(
    train: pd.DataFrame,
    subset: str,
    *,
    window: int = config.WINDOW,
    cap: int = config.RUL_CAP,
    max_iter: int = 300,
    extra_metadata: dict[str, Any] | None = None,
) -> ModelBundle:
    """Fit the scaler and the three models on raw training data."""
    labelled = add_rul(train, cap=cap)
    scaler = RegimeScaler(n_regimes=config.N_REGIMES[subset]).fit(labelled)
    feats = build_features(scaler.transform(labelled), scaler.kept_sensors, window)
    feature_cols = list(feats.columns)
    y = labelled["rul"]

    point = _regressor(max_iter).fit(feats[feature_cols], y)
    lower = _regressor(max_iter, loss="quantile", quantile=config.LOWER_QUANTILE)
    upper = _regressor(max_iter, loss="quantile", quantile=config.UPPER_QUANTILE)
    lower.fit(feats[feature_cols], y)
    upper.fit(feats[feature_cols], y)

    ranges = {
        col: (float(train[col].min()), float(train[col].max()))
        for col in [*SETTING_COLS, *SENSOR_COLS]
    }
    trained_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "subset": subset,
        "model_version": f"{subset}-hgb-{trained_at:%Y%m%d-%H%M%S}",
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "n_train_engines": int(train["unit"].nunique()),
        "sensors_kept": scaler.kept_sensors,
        "window": window,
        "rul_cap": cap,
        "interval": f"{config.LOWER_QUANTILE:.0%}-{config.UPPER_QUANTILE:.0%} quantiles",
    }
    metadata.update(extra_metadata or {})
    return ModelBundle(
        scaler=scaler,
        point_model=point,
        lower_model=lower,
        upper_model=upper,
        feature_cols=feature_cols,
        window=window,
        rul_cap=cap,
        input_ranges=ranges,
        metadata=metadata,
    )


def evaluate_bundle(
    bundle: ModelBundle, test: pd.DataFrame, test_rul: pd.DataFrame
) -> dict[str, float]:
    """Score the bundle on the held-out test engines (one prediction per engine)."""
    pred = bundle.predict_last(test)
    truth = test_rul.set_index("unit")["rul"].sort_index()
    if not pred.index.equals(truth.index):
        raise ValueError("test engines and RUL file do not line up")
    y_true = truth.to_numpy(dtype=float)
    y_cap = np.minimum(y_true, bundle.rul_cap)  # the model cannot express more than the cap
    inside = (y_cap >= pred["rul_lower"].to_numpy()) & (y_cap <= pred["rul_upper"].to_numpy())
    return {
        "test_rmse": rmse(y_true, pred["rul"]),
        "test_nasa_score": nasa_score(y_true, pred["rul"]),
        "interval_coverage": float(inside.mean()),
        "mean_interval_width": float((pred["rul_upper"] - pred["rul_lower"]).mean()),
    }


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    """Load a bundle. Only load files you created yourself: joblib files can run code."""
    obj = joblib.load(path)
    if not isinstance(obj, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return obj
