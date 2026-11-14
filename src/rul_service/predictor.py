"""Validate raw readings and turn them into a RUL prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rul_service import config
from rul_service.data import SENSOR_COLS, SETTING_COLS
from rul_service.model import ModelBundle
from rul_service.utils import timed

Status = Literal["healthy", "watch", "critical"]
REQUIRED = ["cycle", *SETTING_COLS, *SENSOR_COLS]


class InputError(ValueError):
    """The request is well-formed JSON but not usable data (the API maps this to HTTP 422)."""


@dataclass(frozen=True)
class Prediction:
    engine_id: str
    rul: float
    rul_lower: float
    rul_upper: float
    status: Status
    model_version: str
    cycles_used: int


def health_status(rul: float) -> Status:
    if rul < config.CRITICAL_BELOW:
        return "critical"
    if rul < config.HEALTHY_ABOVE:
        return "watch"
    return "healthy"


class Predictor:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def validate(self, readings: Sequence[Mapping[str, float]]) -> pd.DataFrame:
        """Check the readings and return them as a DataFrame in training column order."""
        if not readings:
            raise InputError("readings must not be empty")
        if len(readings) > config.MAX_READINGS:
            raise InputError(f"at most {config.MAX_READINGS} readings are accepted")
        missing = sorted({c for r in readings for c in REQUIRED if c not in r})
        if missing:
            raise InputError(f"missing fields: {missing}")

        df = pd.DataFrame([{c: r[c] for c in REQUIRED} for r in readings]).astype(float)
        if not np.isfinite(df.to_numpy()).all():
            raise InputError("readings contain NaN or infinite values")

        cycles = df["cycle"].to_numpy()
        if cycles[0] < 1 or (np.diff(cycles) != 1).any():
            raise InputError("cycle numbers must start at 1 or later and rise by exactly 1")
        if len(df) < self.bundle.window and cycles[0] != 1:
            raise InputError(
                f"send at least {self.bundle.window} consecutive cycles, "
                "or the engine's full history starting at cycle 1"
            )

        outside = []
        for col, (lo, hi) in self.bundle.input_ranges.items():
            tol = config.RANGE_MARGIN * (hi - lo) + 1e-6 * max(1.0, abs(lo), abs(hi))
            if ((df[col] < lo - tol) | (df[col] > hi + tol)).any():
                outside.append(col)
        if outside:
            raise InputError(f"values far outside the training range for: {outside}")

        df["cycle"] = df["cycle"].astype(int)
        df.insert(0, "unit", 1)
        return df

    @timed
    def predict(self, engine_id: str, readings: Sequence[Mapping[str, float]]) -> Prediction:
        df = self.validate(readings)
        row = self.bundle.predict_last(df).iloc[0]
        rul = round(float(row["rul"]), 1)
        return Prediction(
            engine_id=engine_id,
            rul=rul,
            rul_lower=round(float(row["rul_lower"]), 1),
            rul_upper=round(float(row["rul_upper"]), 1),
            status=health_status(rul),
            model_version=self.bundle.version,
            cycles_used=len(df),
        )
