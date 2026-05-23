"""Tests for impute_ts: strategies, limit, edge cases, error handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.timeseries import impute_ts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_ts() -> pd.DataFrame:
    """Small numeric + categorical DataFrame with a DatetimeIndex.

    Gap layout for 'temp'  (length 2): rows 1, 2
    Gap layout for 'status' (length 2): rows 4, 5
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "temp":   [10.0, np.nan, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0],
            "status": ["ok", "ok",  "ok",  "ok",  None,  None, "ok", "ok"],
        },
        index=idx,
    )


@pytest.fixture()
def long_gap_ts() -> pd.DataFrame:
    """Numeric series with a 5-element gap — used to test limit behaviour."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
    return pd.DataFrame({"val": values}, index=idx)


@pytest.fixture()
def integer_index_ts() -> pd.DataFrame:
    """DataFrame with a plain RangeIndex — used to check that method='time' raises."""
    return pd.DataFrame(
        {"val": [1.0, np.nan, 3.0, 4.0, 5.0]},
        index=pd.RangeIndex(5),
    )


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestStrategies:
    """Each strategy must produce a fully-imputed numeric column."""

    @pytest.mark.parametrize("method", ["ffill", "bfill", "linear", "time", "spline"])
    def test_no_nan_after_imputation(self, simple_ts, method):
        result = impute_ts(simple_ts[["temp"]], method=method)
        assert result["temp"].isnull().sum() == 0

    @pytest.mark.parametrize("method", ["ffill", "bfill", "linear", "time", "spline"])
    def test_returns_dataframe(self, simple_ts, method):
        result = impute_ts(simple_ts[["temp"]], method=method)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("method", ["ffill", "bfill", "linear", "time", "spline"])
    def test_shape_preserved(self, simple_ts, method):
        original_shape = simple_ts[["temp"]].shape
        result = impute_ts(simple_ts[["temp"]], method=method)
        assert result.shape == original_shape

    def test_ffill_fills_forward(self, simple_ts):
        result = impute_ts(simple_ts[["temp"]], method="ffill")
        # rows 1, 2 should be filled with 10.0 (the last known value)
        assert result["temp"].iloc[1] == pytest.approx(10.0)
        assert result["temp"].iloc[2] == pytest.approx(10.0)

    def test_bfill_fills_backward(self, simple_ts):
        result = impute_ts(simple_ts[["temp"]], method="bfill")
        # rows 1, 2 should be filled with 13.0 (the next known value)
        assert result["temp"].iloc[1] == pytest.approx(13.0)
        assert result["temp"].iloc[2] == pytest.approx(13.0)

    def test_linear_midpoint(self, simple_ts):
        result = impute_ts(simple_ts[["temp"]], method="linear")
        # linear interp between 10.0 (row 0) and 13.0 (row 3)
        # row 1 = 10 + 1*(13-10)/3 = 11.0
        assert result["temp"].iloc[1] == pytest.approx(11.0, abs=0.1)


# ---------------------------------------------------------------------------
# method='time' requires DatetimeIndex
# ---------------------------------------------------------------------------

class TestTimeMethod:
    def test_raises_on_non_datetime_index(self, integer_index_ts):
        """method='time' must raise ValueError for non-DatetimeIndex."""
        with pytest.raises((ValueError, TypeError)):
            impute_ts(integer_index_ts, method="time")

    def test_works_on_datetime_index(self, simple_ts):
        result = impute_ts(simple_ts[["temp"]], method="time")
        assert result["temp"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# limit parameter
# ---------------------------------------------------------------------------

class TestLimit:
    def test_limit_leaves_remaining_nan(self, long_gap_ts):
        """With limit=2 on a gap of 5, rows 3-5 must remain NaN."""
        result = impute_ts(long_gap_ts, method="ffill", limit=2)
        # rows 1, 2 filled; rows 3, 4, 5 still NaN
        assert result["val"].iloc[1] == pytest.approx(1.0)
        assert result["val"].iloc[2] == pytest.approx(1.0)
        assert result["val"].isnull().sum() == 3

    def test_limit_none_fills_all(self, long_gap_ts):
        result = impute_ts(long_gap_ts, method="ffill", limit=None)
        assert result["val"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# columns subset
# ---------------------------------------------------------------------------

class TestColumnsSubset:
    def test_only_specified_column_imputed(self, simple_ts):
        result = impute_ts(simple_ts, method="linear", columns=["temp"])
        assert result["temp"].isnull().sum() == 0
        # 'status' is categorical — impute_ts skips it when columns not specified
        # but here we explicitly only asked for 'temp'
        assert result["status"].isnull().sum() == simple_ts["status"].isnull().sum()

    def test_missing_column_raises(self, simple_ts):
        with pytest.raises(KeyError):
            impute_ts(simple_ts, method="linear", columns=["nonexistent"])


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

class TestInvalidMethod:
    def test_invalid_method_raises_value_error(self, simple_ts):
        with pytest.raises(ValueError, match="method must be one of"):
            impute_ts(simple_ts, method="kalman")
