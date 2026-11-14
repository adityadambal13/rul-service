"""Request and response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from rul_service import config


class CycleReading(BaseModel):
    """One operating cycle: 3 operational settings and 21 sensor values."""

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1, description="Operating cycle number of this reading")
    setting_1: FiniteFloat
    setting_2: FiniteFloat
    setting_3: FiniteFloat
    s_1: FiniteFloat
    s_2: FiniteFloat
    s_3: FiniteFloat
    s_4: FiniteFloat
    s_5: FiniteFloat
    s_6: FiniteFloat
    s_7: FiniteFloat
    s_8: FiniteFloat
    s_9: FiniteFloat
    s_10: FiniteFloat
    s_11: FiniteFloat
    s_12: FiniteFloat
    s_13: FiniteFloat
    s_14: FiniteFloat
    s_15: FiniteFloat
    s_16: FiniteFloat
    s_17: FiniteFloat
    s_18: FiniteFloat
    s_19: FiniteFloat
    s_20: FiniteFloat
    s_21: FiniteFloat


class PredictRequest(BaseModel):
    engine_id: str = Field(min_length=1, max_length=64)
    readings: list[CycleReading] = Field(
        min_length=1,
        max_length=config.MAX_READINGS,
        description="Consecutive cycles, oldest first. Send the latest 30 (or the full history).",
    )


class PredictResponse(BaseModel):
    engine_id: str
    rul: float = Field(description="Predicted remaining useful life, in cycles")
    rul_lower: float = Field(description="Lower end of the ~80% interval")
    rul_upper: float = Field(description="Upper end of the ~80% interval")
    status: Literal["healthy", "watch", "critical"]
    model_version: str
    cycles_used: int


class ModelInfo(BaseModel):
    model_version: str
    subset: str
    trained_at: str
    window: int
    rul_cap: int
    interval: str
    n_train_engines: int
    sensors_kept: list[str]
