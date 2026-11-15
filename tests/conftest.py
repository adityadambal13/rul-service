"""Shared fixtures: a small synthetic dataset and a model bundle trained on it."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rul_service.main import create_app
from rul_service.model import ModelBundle, fit_bundle, save_bundle
from rul_service.synthetic import make_engine, make_training_frame, readings_from

__all__ = ["readings_from"]  # re-exported so tests can import it from here


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    return make_training_frame(n_engines=25, seed=0)


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
