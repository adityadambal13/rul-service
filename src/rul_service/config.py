"""Central settings for training and serving."""

from __future__ import annotations

import os
from pathlib import Path

# Number of operating regimes per C-MAPSS subset (used to scale sensors per regime).
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

WINDOW = 30  # cycles of history used for window features
RUL_CAP = 125  # training labels above this are treated as "healthy"

# Intervals come from quantile models: the 10th and 90th percentile -> an 80% band.
LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9

# Illustrative health bands in cycles. Real thresholds come from maintenance planning.
HEALTHY_ABOVE = 60.0
CRITICAL_BELOW = 30.0

MAX_READINGS = 500  # largest request we accept
RANGE_MARGIN = 0.5  # how far beyond the training range (as a fraction) before we reject


def model_path() -> Path:
    """Where the serving app looks for the model bundle (override with RUL_MODEL_PATH)."""
    return Path(os.environ.get("RUL_MODEL_PATH", "models/rul_bundle.joblib"))
