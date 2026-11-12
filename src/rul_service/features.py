"""Regime-aware scaling and window features."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from rul_service.data import SENSOR_COLS, SETTING_COLS


class RegimeScaler:
    """Standardise sensors separately within each operating regime.

    FD001/FD003 have one operating condition; FD002/FD004 have six. Without per-regime
    scaling the operating condition swamps the degradation signal.
    """

    def __init__(self, n_regimes: int = 1, random_state: int = 0) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._kmeans: KMeans | None = None
        self._mean: pd.DataFrame | None = None
        self._std: pd.DataFrame | None = None
        self.kept_sensors: list[str] = []

    def _regime(self, df: pd.DataFrame) -> pd.Series:
        if self._kmeans is None:
            return pd.Series(0, index=df.index)
        labels = self._kmeans.predict(df[SETTING_COLS].to_numpy())
        return pd.Series(labels, index=df.index)

    def fit(self, df: pd.DataFrame) -> RegimeScaler:
        if self.n_regimes > 1:
            self._kmeans = KMeans(
                n_clusters=self.n_regimes, n_init=10, random_state=self.random_state
            ).fit(df[SETTING_COLS].to_numpy())
        regime = self._regime(df)
        grouped = df[SENSOR_COLS].groupby(regime)
        self._mean = grouped.mean()
        std = grouped.std(ddof=0)
        self._std = std.where(std > 1e-8, 1.0)
        scaled = self.transform(df)
        self.kept_sensors = [c for c in SENSOR_COLS if scaled[c].std() > 1e-6]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._mean is None or self._std is None:
            raise RuntimeError("RegimeScaler must be fitted before transform")
        regime = self._regime(df)
        out = df.copy()
        mean = self._mean.loc[regime].set_index(df.index)
        std = self._std.loc[regime].set_index(df.index)
        out[SENSOR_COLS] = (df[SENSOR_COLS] - mean) / std
        return out


def build_features(df: pd.DataFrame, sensor_cols: list[str], window: int = 30) -> pd.DataFrame:
    """Per-row features using only the current and previous cycles of the same engine.

    For each sensor: the current value, the mean over the last `window` cycles, and the
    change versus `window - 1` cycles ago (the engine's first reading is used while history
    is shorter than the window). Also includes the engine's age in cycles.
    """
    feats = pd.DataFrame(index=df.index)
    feats["cycle"] = df["cycle"]
    grouped = df.groupby("unit")
    for col in sensor_cols:
        s = grouped[col]
        rolling_mean = s.transform(lambda x: x.rolling(window, min_periods=1).mean())
        lagged = s.shift(window - 1).fillna(s.transform("first"))
        feats[f"{col}_last"] = df[col]
        feats[f"{col}_mean"] = rolling_mean
        feats[f"{col}_delta"] = df[col] - lagged
    return feats


def last_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting each engine's final recorded cycle."""
    return df["cycle"] == df.groupby("unit")["cycle"].transform("max")
