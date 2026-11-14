"""Shared fixtures: a small synthetic dataset and a model bundle trained on it."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rul_service.main import create_app
from rul_service.model import ModelBundle, fit_bundle, save_bundle

CONSTANT_SENSORS = (1, 5)  # exactly constant, like the dead sensors in the real data


def make_engine(rng: np.random.Generator, unit: int, n_cycles: int) -> pd.DataFrame:
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


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.concat(
        [make_engine(rng, u, int(rng.integers(120, 200))) for u in range(1, 26)],
        ignore_index=True,
    )


@pytest.fixture(scope="session")
def engine_df() -> pd.DataFrame:
    """One longer engine, unseen in training, used to test serving."""
    return make_engine(np.random.default_rng(99), unit=1, n_cycles=150)


@pytest.fixture(scope="session")
def bundle(train_df: pd.DataFrame) -> ModelBundle:
    return fit_bundle(train_df, "FD001", max_iter=40)


@pytest.fixture
def bundle_path(bundle: ModelBundle, tmp_path: Path) -> Path:
    path = tmp_path / "bundle.joblib"
    save_bundle(bundle, path)
    return path


@pytest.fixture
def client(bundle_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(bundle_path)) as c:  # the `with` runs startup (model load)
        yield c


@pytest.fixture
def client_without_model(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        yield c


def readings_from(df: pd.DataFrame, last_cycle: int, n: int = 30) -> list[dict[str, float]]:
    """The `n` readings ending at `last_cycle`, in the shape the API expects."""
    window = df[(df["cycle"] > last_cycle - n) & (df["cycle"] <= last_cycle)]
    cols = [c for c in window.columns if c != "unit"]
    return [{c: float(v) for c, v in row.items()} for row in window[cols].to_dict("records")]
