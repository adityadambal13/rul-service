"""Synthetic engines in the C-MAPSS column layout, for tests and CI smoke checks.

This is NOT real data. It exists so tests, demos and the CI pipeline can exercise the
whole stack without downloading the NASA files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSTANT_SENSORS = (1, 5)  # exactly constant, like the dead sensors in the real data


def make_engine(rng: np.random.Generator, unit: int, n_cycles: int) -> pd.DataFrame:
    """One engine that degrades smoothly over `n_cycles` cycles."""
    cycles = np.arange(1, n_cycles + 1)
    degradation = (cycles / n_cycles) ** 2
    data: dict[str, object] = {"unit": unit, "cycle": cycles}
    for i in (1, 2):
        data[f"setting_{i}"] = rng.normal(0, 0.001, n_cycles)
    data["setting_3"] = 100.0
    for k in range(1, 22):
        base = 100.0 + 10 * k
        if k in CONSTANT_SENSORS:
            data[f"s_{k}"] = base
        else:
            slope = (1 if k % 2 else -1) * k
            data[f"s_{k}"] = base + slope * degradation + rng.normal(0, 0.3, n_cycles)
    return pd.DataFrame(data)


def make_training_frame(n_engines: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat(
        [make_engine(rng, u, int(rng.integers(120, 200))) for u in range(1, n_engines + 1)],
        ignore_index=True,
    )


def readings_from(df: pd.DataFrame, last_cycle: int, n: int = 30) -> list[dict[str, float]]:
    """The `n` readings ending at `last_cycle`, in the shape the API expects."""
    window = df[(df["cycle"] > last_cycle - n) & (df["cycle"] <= last_cycle)]
    cols = [c for c in window.columns if c != "unit"]
    return [{str(c): float(v) for c, v in row.items()} for row in window[cols].to_dict("records")]
