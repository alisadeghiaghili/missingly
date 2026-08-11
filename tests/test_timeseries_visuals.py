"""Tests for vis_ts_miss, vis_gap_lengths, vis_miss_over_time."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless

from missingly.timeseries import vis_ts_miss, vis_gap_lengths, vis_miss_over_time


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


@pytest.fixture()
def ts_df() -> pd.DataFrame:
    """Small time-indexed DataFrame with gaps in two columns."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 2))
    data[3, 0] = np.nan
    data[4, 0] = np.nan
    data[10, 1] = np.nan
    return pd.DataFrame(data, index=idx, columns=["temp", "humidity"])


@pytest.fixture()
def no_gap_df() -> pd.DataFrame:
    """DataFrame with no missing values — used for vis_gap_lengths edge case."""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({"x": np.arange(10, dtype=float)}, index=idx)


# ---------------------------------------------------------------------------
# vis_ts_miss
# ---------------------------------------------------------------------------

class TestVisTsMiss:
    def test_returns_axes(self, ts_df):
        ax = vis_ts_miss(ts_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_no_exception(self, ts_df):
        vis_ts_miss(ts_df)  # must not raise

    def test_title_set(self, ts_df):
        ax = vis_ts_miss(ts_df, title="My Title")
        assert ax.get_title() == "My Title"

    def test_accepts_existing_axes(self, ts_df):
        fig, ax_in = plt.subplots()
        ax_out = vis_ts_miss(ts_df, ax=ax_in)
        assert ax_out is ax_in


# ---------------------------------------------------------------------------
# vis_gap_lengths
# ---------------------------------------------------------------------------

class TestVisGapLengths:
    def test_returns_figure_hist(self, ts_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig = vis_gap_lengths(ts_df, kind="hist")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_box(self, ts_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig = vis_gap_lengths(ts_df, kind="box")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_exception_hist(self, ts_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            vis_gap_lengths(ts_df, kind="hist")

    def test_no_gap_returns_figure_with_message(self, no_gap_df):
        """When there are no gaps the function should still return a Figure (not raise)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig = vis_gap_lengths(no_gap_df, kind="box")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_emits_future_warning(self, ts_df):
        with pytest.warns(FutureWarning):
            vis_gap_lengths(ts_df)


# ---------------------------------------------------------------------------
# vis_miss_over_time
# ---------------------------------------------------------------------------

class TestVisMissOverTime:
    def test_returns_axes(self, ts_df):
        ax = vis_miss_over_time(ts_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_no_exception(self, ts_df):
        vis_miss_over_time(ts_df)

    def test_ylim_0_100(self, ts_df):
        ax = vis_miss_over_time(ts_df)
        assert ax.get_ylim()[0] == pytest.approx(0)
        assert ax.get_ylim()[1] == pytest.approx(100)

    def test_accepts_existing_axes(self, ts_df):
        fig, ax_in = plt.subplots()
        ax_out = vis_miss_over_time(ts_df, ax=ax_in)
        assert ax_out is ax_in

    def test_custom_window(self, ts_df):
        ax = vis_miss_over_time(ts_df, window=5)
        assert "window=5" in ax.get_ylabel()
