"""RUL labels for run-to-failure training data."""

from __future__ import annotations

import pandas as pd

DEFAULT_RUL_CAP = 125


def add_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add a 'rul' column: cycles remaining until each engine's last recorded cycle.

    Early in an engine's life degradation is invisible in the sensors, so the target
    is commonly capped (piece-wise linear RUL). Pass cap=None to keep raw values.
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last_cycle - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out
