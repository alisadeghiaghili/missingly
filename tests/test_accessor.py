"""Tests for the pandas DataFrame accessor (df.miss.*)."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

import missingly  # noqa: F401 — registers the accessor


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


@pytest.fixture
def df():
    """Small DataFrame with NaN values for accessor tests."""
    return pd.DataFrame({
        "A": [1.0, np.nan, 3.0, 4.0],
        "B": [np.nan, 2.0, np.nan, 4.0],
        "C": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def full_df():
    """Fully-observed DataFrame (no NaNs) for zero-missing edge cases."""
    return pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})


# ---------------------------------------------------------------------------
# Accessor registration
# ---------------------------------------------------------------------------

def test_accessor_registered(df):
    """The 'miss' namespace must be accessible on any DataFrame."""
    assert hasattr(df, "miss")


def test_accessor_type_error():
    """Accessing .miss on a non-DataFrame raises TypeError."""
    from missingly.accessor import MissinglyAccessor
    with pytest.raises(TypeError):
        MissinglyAccessor("not a dataframe")


# ---------------------------------------------------------------------------
# Summary methods
# ---------------------------------------------------------------------------

def test_n_miss(df):
    """n_miss returns the correct total missing count."""
    assert df.miss.n_miss() == 3


def test_n_complete(df):
    """n_complete returns total cells minus missing cells."""
    assert df.miss.n_complete() == df.size - 3


def test_pct_miss(df):
    """pct_miss is between 0 and 100."""
    pct = df.miss.pct_miss()
    assert 0.0 <= pct <= 100.0


def test_pct_complete(df):
    """pct_complete + pct_miss should sum to 100."""
    assert abs(df.miss.pct_miss() + df.miss.pct_complete() - 100.0) < 1e-6


def test_miss_var_summary_shape(df):
    """miss_var_summary returns a DataFrame with one row per column."""
    result = df.miss.miss_var_summary()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df.columns)


def test_miss_case_summary_shape(df):
    """miss_case_summary returns a DataFrame with one row per DataFrame row."""
    result = df.miss.miss_case_summary()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_bind_shadow_columns(df):
    """bind_shadow doubles the column count with _NA suffix columns."""
    result = df.miss.bind_shadow()
    assert result.shape[1] == df.shape[1] * 2


# ---------------------------------------------------------------------------
# Manipulation — chaining
# ---------------------------------------------------------------------------

def test_replace_with_na_chain(df):
    """replace_with_na returns a DataFrame (chainable)."""
    result = df.miss.replace_with_na({"C": 3.0})
    assert isinstance(result, pd.DataFrame)
    assert pd.isnull(result.loc[2, "C"])


def test_replace_with_na_all_chain(df):
    """replace_with_na_all returns a DataFrame (chainable)."""
    result = df.miss.replace_with_na_all(lambda x: x == 1.0)
    assert isinstance(result, pd.DataFrame)


def test_clean_names_chain():
    """clean_names returns a DataFrame with normalised column names."""
    d = pd.DataFrame(columns=["First Name", "Last Name"])
    result = d.miss.clean_names()
    assert result.columns.tolist() == ["first_name", "last_name"]


def test_remove_empty_chain(df):
    """remove_empty returns a DataFrame (chainable)."""
    result = df.miss.remove_empty()
    assert isinstance(result, pd.DataFrame)


def test_coalesce_chain(df):
    """coalesce_columns fills NaN in target from donor and stays chainable."""
    result = df.miss.coalesce_columns("A", "B")
    assert isinstance(result, pd.DataFrame)
    assert not pd.isnull(result.loc[1, "A"])  # was NaN, filled from B


def test_miss_as_feature_chain(df):
    """miss_as_feature returns a DataFrame with _NA indicator columns."""
    result = df.miss.miss_as_feature()
    assert isinstance(result, pd.DataFrame)
    assert "A_NA" in result.columns
    assert "B_NA" in result.columns
    assert "C_NA" not in result.columns  # C has no missing


def test_full_chain(df):
    """Multiple manipulation methods can be chained together."""
    result = (
        df
        .miss.replace_with_na({"C": 99.0})
        .miss.remove_empty(thresh_col=0.99)
        .miss.miss_as_feature()
    )
    assert isinstance(result, pd.DataFrame)
    # Original df must not be mutated
    assert "A_NA" not in df.columns


# ---------------------------------------------------------------------------
# Imputation methods
# ---------------------------------------------------------------------------

def test_impute_mean_chain(df):
    """impute_mean returns a fully-observed DataFrame."""
    result = df.miss.impute_mean()
    assert isinstance(result, pd.DataFrame)
    assert result.isnull().sum().sum() == 0


def test_impute_median_chain(df):
    """impute_median returns a fully-observed DataFrame."""
    result = df.miss.impute_median()
    assert result.isnull().sum().sum() == 0


def test_impute_mode_chain(df):
    """impute_mode returns a fully-observed DataFrame."""
    result = df.miss.impute_mode()
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Visualisation methods
# ---------------------------------------------------------------------------

def test_vis_matrix_returns_axes(df):
    """matrix() returns a matplotlib Axes."""
    import matplotlib.axes
    ax = df.miss.matrix()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_vis_bar_returns_axes(df):
    """bar() returns a matplotlib Axes."""
    import matplotlib.axes
    ax = df.miss.bar()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_vis_heatmap_returns_axes(df):
    """heatmap() returns a matplotlib Axes."""
    import matplotlib.axes
    ax = df.miss.heatmap()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_vis_miss_returns_axes(df):
    """vis_miss() returns a matplotlib Axes."""
    import matplotlib.axes
    ax = df.miss.vis_miss()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_vis_miss_var_pct_returns_axes(df):
    """miss_var_pct() returns a matplotlib Axes."""
    import matplotlib.axes
    ax = df.miss.miss_var_pct()
    assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Original df not mutated
# ---------------------------------------------------------------------------

def test_accessor_does_not_mutate(df):
    """No accessor method should mutate the original DataFrame."""
    original_cols = list(df.columns)
    original_vals = df.copy()

    df.miss.replace_with_na({"A": 1.0})
    df.miss.remove_empty()
    df.miss.miss_as_feature()
    df.miss.clean_names()

    assert list(df.columns) == original_cols
    pd.testing.assert_frame_equal(df, original_vals)
