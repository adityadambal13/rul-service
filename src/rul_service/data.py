"""Load NASA C-MAPSS subsets into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS  # 26 columns


def load_subset(raw_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, test_rul) for one subset such as 'FD001'."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    train = pd.read_csv(raw_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(raw_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None, names=["rul"])
    rul.insert(0, "unit", range(1, len(rul) + 1))
    return train, test, rul
