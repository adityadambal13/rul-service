from pathlib import Path

import pytest

from rul_service.data import COLUMNS, SUBSETS, load_subset

RAW = Path("data/raw")
EXPECTED_TRAIN_UNITS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw data not downloaded")


@pytest.mark.parametrize("subset", SUBSETS)
def test_subset_is_well_formed(subset: str) -> None:
    train, test, rul = load_subset(RAW, subset)
    assert list(train.columns) == COLUMNS
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert train["unit"].nunique() == EXPECTED_TRAIN_UNITS[subset]
    assert len(rul) == test["unit"].nunique()
    # cycles for each engine must run 1..n with no gaps
    counts = train.groupby("unit")["cycle"].agg(["min", "max", "count"])
    assert (counts["min"] == 1).all()
    assert (counts["max"] == counts["count"]).all()
