"""Tests for missingly.timeseries module.

Covers:
- miss_ts_summary: shape, values, no-missing case, all-missing case
- gap_table: tidy structure, correct gap detection, no-gaps case
- vis_ts_miss / vis_gap_lengths / vis_miss_over_time: return correct Axes type
- impute_ts: all five strategies, limit enforcement, categorical ffill,
  sentinel support, unsorted DatetimeIndex raises, invalid strategy raises,
  strategy='time' on non-DatetimeIndex raises, mutation safety
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missingly.timeseries import (
    miss_ts_summary,
    gap_table,
    vis_ts_miss,
    vis_gap_lengths,
    vis_miss_over_time,
    impute_ts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_df():
    """Small datetime-indexed DataFrame with two columns and known gaps."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "temp":  [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "humid": [80.0, 81.0, np.nan, 83.0, np.nan, np.nan, np.nan, 87.0, 88.0, 89.0],
        },
        index=dates,
    )


@pytest.fixture
def ts_df_mixed(ts_df):
    """ts_df augmented with a categorical column."""
    result = ts_df.copy()
    result["status"] = ["ok", None, None, "ok", "warn", None, "ok", "ok", "ok", "ok"]
    return result


@pytest.fixture
def int_idx_df():
    """DataFrame with integer (ordinal) index."""
    return pd.DataFrame(
        {"val": [1.0, np.nan, np.nan, 4.0, 5.0]},
        index=range(5),
    )


# ---------------------------------------------------------------------------
# miss_ts_summary
# ---------------------------------------------------------------------------

class TestMissTsSummary:
    def test_shape(self, ts_df):
        """Summary has one row per column."""
        summary = miss_ts_summary(ts_df)
        assert summary.shape[0] == ts_df.shape[1]

    def test_n_miss_correct(self, ts_df):
        """n_miss matches actual missing count."""
        summary = miss_ts_summary(ts_df)
        assert summary.loc["temp", "n_miss"] == int(ts_df["temp"].isnull().sum())
        assert summary.loc["humid", "n_miss"] == int(ts_df["humid"].isnull().sum())

    def test_n_gaps_temp(self, ts_df):
        """temp has exactly 2 gaps."""
        summary = miss_ts_summary(ts_df)
        assert summary.loc["temp", "n_gaps"] == 2

    def test_max_gap_len(self, ts_df):
        """humid has a max gap of 3 consecutive rows."""
        summary = miss_ts_summary(ts_df)
        assert summary.loc["humid", "max_gap_len"] == 3

    def test_no_missing(self, ts_df):
        """Columns with no missing values get zero counts."""
        df_clean = ts_df.dropna()
        summary = miss_ts_summary(df_clean)
        assert (summary["n_miss"] == 0).all()
        assert (summary["n_gaps"] == 0).all()

    def test_all_missing(self):
        """Entirely missing column is treated as a single gap."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan, np.nan]}, index=dates)
        summary = miss_ts_summary(df)
        assert summary.loc["x", "n_gaps"] == 1
        assert summary.loc["x", "max_gap_len"] == 4

    def test_sentinel_support(self):
        """Values in missing_values list are treated as missing."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [1.0, -99.0, -99.0, 4.0]}, index=dates)
        summary = miss_ts_summary(df, missing_values=[-99])
        assert summary.loc["x", "n_miss"] == 2
        assert summary.loc["x", "n_gaps"] == 1


# ---------------------------------------------------------------------------
# gap_table
# ---------------------------------------------------------------------------

class TestGapTable:
    def test_columns(self, ts_df):
        """gap_table has required columns."""
        gt = gap_table(ts_df)
        for col in ["column", "gap_id", "start", "end", "length"]:
            assert col in gt.columns

    def test_temp_has_two_gaps(self, ts_df):
        """temp column has 2 gaps."""
        gt = gap_table(ts_df)
        assert len(gt[gt["column"] == "temp"]) == 2

    def test_gap_lengths_correct(self, ts_df):
        """Gap lengths for temp are 2 and 1."""
        gt = gap_table(ts_df)
        temp_gaps = gt[gt["column"] == "temp"]["length"].tolist()
        assert sorted(temp_gaps) == [1, 2]

    def test_no_gaps_returns_empty(self, ts_df):
        """DataFrame without missing returns empty table."""
        df_clean = ts_df.ffill().bfill()
        gt = gap_table(df_clean)
        assert gt.empty

    def test_filter_long_gaps(self, ts_df):
        """Filtering gap_table for length >= 3 returns only humid's 3-row gap."""
        gt = gap_table(ts_df)
        big = gt[gt["length"] >= 3]
        assert len(big) == 1
        assert big.iloc[0]["column"] == "humid"


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

class TestVisualisations:
    def test_vis_ts_miss_returns_axes(self, ts_df):
        ax = vis_ts_miss(ts_df)
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_gap_lengths_hist_returns_axes(self, ts_df):
        ax = vis_gap_lengths(ts_df, kind="hist")
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_gap_lengths_box_returns_axes(self, ts_df):
        ax = vis_gap_lengths(ts_df, kind="box")
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_gap_lengths_no_gaps_raises(self, ts_df):
        df_clean = ts_df.ffill().bfill()
        with pytest.raises(ValueError, match="No gaps"):
            vis_gap_lengths(df_clean)

    def test_vis_miss_over_time_returns_axes(self, ts_df):
        ax = vis_miss_over_time(ts_df, window=3)
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_ts_miss_integer_index(self, int_idx_df):
        """vis_ts_miss works with integer index."""
        ax = vis_ts_miss(int_idx_df)
        assert hasattr(ax, "get_title")
        plt.close("all")


# ---------------------------------------------------------------------------
# impute_ts
# ---------------------------------------------------------------------------

class TestImpuseTs:
    @pytest.mark.parametrize("strategy", ["ffill", "bfill", "linear", "time", "spline"])
    def test_no_missing_after_impute(self, ts_df, strategy):
        """All strategies fully impute the numeric columns."""
        result = impute_ts(ts_df, strategy=strategy)
        assert result[ts_df.select_dtypes(include=[np.number]).columns].isnull().sum().sum() == 0

    def test_ffill_value_correct(self):
        """ffill propagates the last observed value forward."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]}, index=dates)
        result = impute_ts(df, strategy="ffill")
        assert result.loc[dates[1], "x"] == 1.0
        assert result.loc[dates[2], "x"] == 1.0

    def test_linear_value_correct(self):
        """linear strategy interpolates the midpoint."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [0.0, np.nan, np.nan, 3.0]}, index=dates)
        result = impute_ts(df, strategy="linear")
        assert abs(result.loc[dates[1], "x"] - 1.0) < 1e-9
        assert abs(result.loc[dates[2], "x"] - 2.0) < 1e-9

    def test_limit_enforced(self):
        """limit=1 leaves second consecutive missing unfilled."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, 5.0]}, index=dates)
        result = impute_ts(df, strategy="ffill", limit=1)
        # First gap filled, second and third remain NaN
        assert result.loc[dates[1], "x"] == 1.0
        assert pd.isna(result.loc[dates[2], "x"])
        assert pd.isna(result.loc[dates[3], "x"])

    def test_categorical_always_ffilled(self, ts_df_mixed):
        """Categorical column is always forward-filled."""
        result = impute_ts(ts_df_mixed, strategy="linear")
        # rows 1 and 2 (index 2024-01-02, 2024-01-03) were None in 'status'
        assert result["status"].iloc[1] == "ok"
        assert result["status"].iloc[2] == "ok"

    def test_integer_index_linear(self, int_idx_df):
        """linear strategy works with integer index."""
        result = impute_ts(int_idx_df, strategy="linear")
        assert result.isnull().sum().sum() == 0

    def test_time_strategy_requires_datetimeindex(self, int_idx_df):
        """strategy='time' raises ValueError on non-DatetimeIndex."""
        with pytest.raises(ValueError, match="DatetimeIndex"):
            impute_ts(int_idx_df, strategy="time")

    def test_invalid_strategy_raises(self, ts_df):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy"):
            impute_ts(ts_df, strategy="interpolate")

    def test_unsorted_index_raises(self):
        """Unsorted DatetimeIndex raises ValueError."""
        dates = pd.date_range("2024-01-05", periods=3, freq="D")
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan]}, index=dates[::-1])
        with pytest.raises(ValueError, match="sorted"):
            impute_ts(df, strategy="linear")

    def test_sentinel_replaced(self):
        """Sentinel values in missing_values are replaced before imputation."""
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"x": [1.0, -99.0, -99.0, 4.0]}, index=dates)
        result = impute_ts(df, strategy="linear", missing_values=[-99])
        assert result.isnull().sum().sum() == 0
        assert abs(result.loc[dates[1], "x"] - 2.0) < 1e-9

    def test_mutation_safety(self, ts_df):
        """impute_ts does not mutate the input DataFrame."""
        original = ts_df.copy()
        impute_ts(ts_df, strategy="linear")
        pd.testing.assert_frame_equal(ts_df, original)
