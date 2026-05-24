"""Tests for miss_ts_summary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.timeseries import miss_ts_summary


@pytest.fixture()
def no_nan_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame({"temp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=idx)


@pytest.fixture()
def two_gap_df() -> pd.DataFrame:
    """Two distinct gaps: length-2 (rows 1-2) and length-3 (rows 5-7)."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, 9.0, 10.0]
    return pd.DataFrame({"temp": values}, index=idx)


class TestNoNan:
    def test_n_miss_zero(self, no_nan_df):
        result = miss_ts_summary(no_nan_df)
        assert result.loc["temp", "n_miss"] == 0

    def test_n_gaps_zero(self, no_nan_df):
        result = miss_ts_summary(no_nan_df)
        assert result.loc["temp", "n_gaps"] == 0

    def test_mean_gap_zero(self, no_nan_df):
        result = miss_ts_summary(no_nan_df)
        assert result.loc["temp", "mean_gap"] == 0.0

    def test_max_gap_zero(self, no_nan_df):
        result = miss_ts_summary(no_nan_df)
        assert result.loc["temp", "max_gap"] == 0


class TestTwoGaps:
    def test_n_miss(self, two_gap_df):
        # 2 + 3 = 5 missing
        result = miss_ts_summary(two_gap_df)
        assert result.loc["temp", "n_miss"] == 5

    def test_n_gaps(self, two_gap_df):
        result = miss_ts_summary(two_gap_df)
        assert result.loc["temp", "n_gaps"] == 2

    def test_max_gap(self, two_gap_df):
        result = miss_ts_summary(two_gap_df)
        assert result.loc["temp", "max_gap"] == 3

    def test_mean_gap(self, two_gap_df):
        # mean of [2, 3] = 2.5
        result = miss_ts_summary(two_gap_df)
        assert result.loc["temp", "mean_gap"] == pytest.approx(2.5)

    def test_returns_dataframe(self, two_gap_df):
        result = miss_ts_summary(two_gap_df)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_column_names(self, two_gap_df):
        result = miss_ts_summary(two_gap_df)
        assert "temp" in result.index

    def test_longest_gap_boundaries(self, two_gap_df):
        """The longest gap (length 3) starts at 2024-01-06 and ends at 2024-01-09 (exclusive end).

        We verify via gap_table that start/end are correct.
        """
        import warnings
        from missingly.timeseries import gap_table

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            gt = gap_table(two_gap_df)

        # gap_table uses 'column' not 'variable'
        longest = gt[gt["column"] == "temp"].sort_values("length", ascending=False).iloc[0]
        assert longest["start"] == pd.Timestamp("2024-01-06")
        # end is the first non-null index *after* the gap
        assert longest["end"] == pd.Timestamp("2024-01-09")
        assert longest["length"] == 3


class TestMultiColumn:
    def test_multiple_columns_in_index(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
             "b": [np.nan, np.nan, np.nan, 4.0, 5.0, 6.0]},
            index=idx,
        )
        result = miss_ts_summary(df)
        assert set(result.index) == {"a", "b"}
        assert result.loc["a", "n_gaps"] == 1
        assert result.loc["b", "n_gaps"] == 1
        assert result.loc["b", "max_gap"] == 3
