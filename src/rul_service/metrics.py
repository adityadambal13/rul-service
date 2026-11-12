"""Evaluation metrics for RUL prediction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """NASA's asymmetric scoring function (lower is better).

    d = predicted - true. Late predictions (d >= 0, engine fails sooner than predicted)
    are penalised more heavily than early ones (d < 0).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))
