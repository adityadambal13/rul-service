"""Train baseline and candidate RUL models and track them with MLflow."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from rul_service.data import load_subset
from rul_service.features import RegimeScaler, build_features, last_rows
from rul_service.labels import DEFAULT_RUL_CAP, add_rul
from rul_service.metrics import nasa_score, rmse

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "rul-cmapss"
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
MODELS = ("ridge_last", "hgb_window")

# MLflow saves scikit-learn models with skops, which refuses types it has not been told
# to trust. We trained this model ourselves, so we trust exactly this one type.
TRUSTED_TYPES = {
    "ridge_last": [],
    "hgb_window": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def _make_model(name: str) -> Ridge | HistGradientBoostingRegressor:
    if name == "ridge_last":
        return Ridge(alpha=1.0)
    if name == "hgb_window":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def _select_columns(features: pd.DataFrame, model_name: str) -> list[str]:
    if model_name == "ridge_last":  # baseline: only the current reading, no history
        return [c for c in features.columns if c.endswith("_last") or c == "cycle"]
    return list(features.columns)


def run(
    subset: str,
    model_name: str,
    window: int = 30,
    cap: int = DEFAULT_RUL_CAP,
    raw_dir: Path = Path("data/raw"),
    tracking_uri: str = TRACKING_URI,
) -> dict[str, float]:
    """Train one model on one subset, log everything to MLflow, return test metrics."""
    train, test, test_rul = load_subset(raw_dir, subset)
    train = add_rul(train, cap=cap)

    scaler = RegimeScaler(n_regimes=N_REGIMES[subset]).fit(train)
    sensors = scaler.kept_sensors
    train_f = build_features(scaler.transform(train), sensors, window)
    test_scaled = scaler.transform(test)
    test_f = build_features(test_scaled, sensors, window)[last_rows(test)]

    cols = _select_columns(train_f, model_name)
    model = _make_model(model_name)
    model.fit(train_f[cols], train["rul"])

    # Test engines are listed in unit order, matching the RUL file.
    order = test.loc[test_f.index, "unit"].argsort()
    y_true = test_rul.sort_values("unit")["rul"].to_numpy()
    y_pred = model.predict(test_f[cols].iloc[order.to_numpy()]).clip(min=0)

    metrics = {"test_rmse": rmse(y_true, y_pred), "test_nasa_score": nasa_score(y_true, y_pred)}

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{subset}-{model_name}"):
        mlflow.log_params(
            {
                "subset": subset,
                "model": model_name,
                "window": window,
                "rul_cap": cap,
                "n_regimes": N_REGIMES[subset],
                "n_features": len(cols),
                "n_sensors_kept": len(sensors),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES[model_name])
    return metrics
