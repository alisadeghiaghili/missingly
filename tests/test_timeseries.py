"""Tests for missingly.timeseries."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from missingly.timeseries import (
    gap_table,
    impute_ts,
    miss_ts_summary,
    vis_gap_lengths,
    vis_miss_over_time,
    vis_ts_miss,
)


@pytest.fixture()
def ts_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {
        "temp": [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        "humid": [1.0, 2.0, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0, 9.0, 10.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def int_idx_df() -> pd.DataFrame:
    data = {
        "a": [1.0, np.nan, 3.0, 4.0, 5.0],
        "b": [np.nan, np.nan, 3.0, 4.0, 5.0],
    }
    return pd.DataFrame(data)


class TestMissTsSummary:
    def test_shape(self, ts_df):
        assert miss_ts_summary(ts_df).shape[0] == 2

    def test_n_miss(self, ts_df):
        result = miss_ts_summary(ts_df)
        assert result.loc["temp", "n_miss"] == 3
        assert result.loc["humid", "n_miss"] == 4

    def test_n_gaps(self, ts_df):
        result = miss_ts_summary(ts_df)
        assert result.loc["temp", "n_gaps"] == 2
        assert result.loc["humid", "n_gaps"] == 2

    def test_max_gap(self, ts_df):
        result = miss_ts_summary(ts_df)
        assert result.loc["temp", "max_gap"] == 2
        assert result.loc["humid", "max_gap"] == 3

    def test_pct_miss_range(self, ts_df):
        result = miss_ts_summary(ts_df)
        assert (result["pct_miss"] >= 0).all()
        assert (result["pct_miss"] <= 100).all()

    def test_unsorted_raises(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")[::-1]
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=idx)
        with pytest.raises(ValueError, match="sorted"):
            miss_ts_summary(df)

    def test_entirely_missing_column(self, ts_df):
        df = ts_df.copy()
        df["all_nan"] = np.nan
        result = miss_ts_summary(df)
        assert result.loc["all_nan", "n_miss"] == len(df)


class TestGapTable:
    def test_columns(self, ts_df):
        gt = gap_table(ts_df)
        for col in ["column", "gap_id", "start", "end", "length"]:
            assert col in gt.columns

    def test_temp_has_two_gaps(self, ts_df):
        gt = gap_table(ts_df)
        assert len(gt[gt["column"] == "temp"]) == 2

    def test_gap_lengths_correct(self, ts_df):
        gt = gap_table(ts_df)
        temp_gaps = gt[gt["column"] == "temp"]["length"].tolist()
        assert sorted(temp_gaps) == [1, 2]

    def test_no_gaps_returns_empty(self, ts_df):
        df_clean = ts_df.ffill().bfill()
        gt = gap_table(df_clean)
        assert gt.empty

    def test_filter_long_gaps(self, ts_df):
        gt = gap_table(ts_df)
        big = gt[gt["length"] >= 3]
        assert len(big) == 1
        assert big.iloc[0]["column"] == "humid"


class TestVisualisations:
    def test_vis_ts_miss_returns_axes(self, ts_df):
        ax = vis_ts_miss(ts_df)
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_gap_lengths_hist_returns_figure(self, ts_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig = vis_gap_lengths(ts_df, kind="hist")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_vis_gap_lengths_box_returns_figure(self, ts_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig = vis_gap_lengths(ts_df, kind="box")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_vis_gap_lengths_no_gaps_raises(self, ts_df):
        df_clean = ts_df.ffill().bfill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(ValueError, match="No gaps"):
                vis_gap_lengths(df_clean)

    def test_vis_miss_over_time_returns_axes(self, ts_df):
        ax = vis_miss_over_time(ts_df, window=3)
        assert hasattr(ax, "get_title")
        plt.close("all")

    def test_vis_ts_miss_integer_index(self, int_idx_df):
        ax = vis_ts_miss(int_idx_df)
        assert hasattr(ax, "get_title")
        plt.close("all")


class TestImpuseTs:
    @pytest.mark.parametrize("strat", ["ffill", "bfill", "linear", "spline"])
    def test_all_strategies(self, ts_df, strat):
        result = impute_ts(ts_df, strat)
        assert result[ts_df.select_dtypes(include=[np.number]).columns].isnull().sum().sum() == 0

    def test_limit_param(self, ts_df):
        result = impute_ts(ts_df, "ffill", limit=1)
        assert result["temp"].isnull().sum() >= 1

    def test_returns_copy(self, ts_df):
        original = ts_df.copy(deep=True)
        impute_ts(ts_df, "ffill")
        pd.testing.assert_frame_equal(ts_df, original)

    def test_no_missing_passthrough(self, ts_df):
        df_clean = ts_df.ffill().bfill()
        result = impute_ts(df_clean, "linear")
        pd.testing.assert_frame_equal(result, df_clean)

    def test_sentinel_values(self, ts_df):
        df = ts_df.copy()
        df["temp"] = df["temp"].fillna(-999)
        result = impute_ts(df, "ffill", missing_values=[-999])
        assert (result["temp"] != -999).all()

    def test_categorical_column(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"num": [1.0, np.nan, 3.0, np.nan, 5.0], "cat": ["a", None, "b", None, "c"]},
            index=idx,
        )
        result = impute_ts(df, "linear")
        assert result["cat"].isnull().sum() == 0

    def test_time_strategy_non_datetime_raises(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            impute_ts(df, "time")

    def test_invalid_strategy_raises(self, ts_df):
        with pytest.raises(ValueError, match="strategy"):
            impute_ts(ts_df, strategy="interpolate")

    def test_unsorted_index_raises(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")[::-1]
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=idx)
        with pytest.raises(ValueError, match="sorted"):
            impute_ts(df, "ffill")
