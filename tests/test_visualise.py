"""Tests for missingly.visualise module.

Conventions
-----------
* Every test closes all open figures before running to avoid state
  leakage between tests.
* Return-type assertions verify the public contract (Axes or dict).
* Title assertions pin the human-readable label so accidental renames
  are caught immediately.
* Edge-case tests (all-missing, no-missing, Persian labels, single
  column) live in clearly named functions.
* Sentinel-value tests use ``missing_values=[-99]`` throughout to
  exercise the ``_nullity`` helper path alongside the NaN path.
"""

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for CI
import matplotlib.pyplot as plt

from missingly import visualise
from missingly import impute
import missingly  # top-level import — validates __init__.py exports


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Numeric dataframe with sentinel missing values (-99)."""
    return pd.DataFrame({
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def nan_df():
    """Dataframe with true NaN missing values across numeric columns."""
    return pd.DataFrame({
        'X': [1.0, np.nan, 3.0, np.nan],
        'Y': [np.nan, 2.0, np.nan, 4.0],
        'Z': [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def persian_df():
    """DataFrame with Persian column names and index labels.

    Verifies that all plotting functions handle Unicode / RTL text
    without raising encoding or rendering errors.
    """
    return pd.DataFrame({
        'درآمد': [1000, np.nan, 3000, np.nan],
        'سن': [25, 30, np.nan, 40],
        'نام': ['علی', None, 'رضا', 'مریم'],
    }, index=['ردیف_۱', 'ردیف_۲', 'ردیف_۳', 'ردیف_۴'])


@pytest.fixture
def all_missing_df():
    """DataFrame where every cell is NaN."""
    return pd.DataFrame({
        'A': [np.nan, np.nan],
        'B': [np.nan, np.nan],
    })


@pytest.fixture
def no_missing_df():
    """DataFrame with no missing values at all."""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
    })


# ---------------------------------------------------------------------------
# Top-level import smoke test
# ---------------------------------------------------------------------------

def test_dendrogram_importable_from_top_level():
    """dendrogram must be accessible via ``missingly.dendrogram``."""
    assert callable(missingly.dendrogram), (
        "missingly.dendrogram is not callable — check __init__.py imports"
    )


def test_new_functions_importable_from_top_level():
    """All new visualisation functions must be importable from the top level."""
    for fn_name in ("heatmap", "vis_miss", "miss_var_pct", "miss_cluster", "miss_which"):
        assert callable(getattr(missingly, fn_name, None)), (
            f"missingly.{fn_name} is not callable — check __init__.py imports"
        )


# ---------------------------------------------------------------------------
# Existing visualisation tests
# ---------------------------------------------------------------------------

def test_matrix(sample_df):
    """matrix() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.matrix(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Data Matrix'


def test_bar(sample_df):
    """bar() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.bar(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Column'


def test_upset(sample_df):
    """upset() returns a dict of Axes without raising."""
    plt.close('all')
    plot = visualise.upset(sample_df, missing_values=[-99])
    plt.close('all')
    assert isinstance(plot, dict)


def test_upset_keys(sample_df):
    """upset() dict must contain exactly the three expected Axes keys."""
    plt.close('all')
    plot = visualise.upset(sample_df, missing_values=[-99])
    plt.close('all')
    assert set(plot.keys()) == {"intersections", "matrix", "totals"}


def test_upset_no_missing(no_missing_df):
    """upset() returns an empty dict when no missing values are present."""
    plt.close('all')
    result = visualise.upset(no_missing_df)
    assert result == {}


def test_scatter_miss(sample_df):
    """scatter_miss() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.scatter_miss(sample_df, x='A', y='B', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Scatter Plot of A vs B with Missing Values'


def test_miss_case(sample_df):
    """miss_case() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.miss_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Case'


def test_vis_impute_dist(sample_df):
    """vis_impute_dist() returns an Axes with the expected title."""
    plt.close('all')
    imputed_df = impute.impute_mean(sample_df)
    ax = visualise.vis_impute_dist(sample_df, imputed_df, 'A')
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Distribution of Original vs. Imputed Data for A'


def test_vis_miss_fct(sample_df):
    """vis_miss_fct() returns an Axes with the expected title."""
    plt.close('all')
    df = sample_df.copy()
    df['Fct'] = ['a', 'b', 'a', 'b']
    ax = visualise.vis_miss_fct(df, 'Fct', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values by Fct'


def test_vis_miss_cumsum_var(sample_df):
    """vis_miss_cumsum_var() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.vis_miss_cumsum_var(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Variable'


def test_vis_miss_cumsum_case(sample_df):
    """vis_miss_cumsum_case() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.vis_miss_cumsum_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Case'


def test_vis_miss_span(sample_df):
    """vis_miss_span() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.vis_miss_span(sample_df, 'A', 2, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values in Spans of 2 for A'


def test_vis_parallel_coords(sample_df):
    """vis_parallel_coords() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.vis_parallel_coords(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Parallel Coordinates Plot of Missingness'


def test_dendrogram(sample_df):
    """dendrogram() returns an Axes with the expected title."""
    plt.close('all')
    ax = visualise.dendrogram(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Dendrogram of Variables by Missing Data Patterns'


# ---------------------------------------------------------------------------
# heatmap() tests
# ---------------------------------------------------------------------------

def test_heatmap_returns_axes(nan_df):
    """heatmap() returns an Axes instance."""
    plt.close('all')
    ax = visualise.heatmap(nan_df)
    assert isinstance(ax, plt.Axes)


def test_heatmap_title(nan_df):
    """heatmap() sets the expected title."""
    plt.close('all')
    ax = visualise.heatmap(nan_df)
    assert ax.get_title() == 'Nullity Correlation Heatmap'


def test_heatmap_sentinel(sample_df):
    """heatmap() works with sentinel missing_values."""
    plt.close('all')
    ax = visualise.heatmap(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_heatmap_no_missing(no_missing_df):
    """heatmap() does not raise when no values are missing."""
    plt.close('all')
    ax = visualise.heatmap(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_heatmap_persian(persian_df):
    """heatmap() handles Persian column names without raising."""
    plt.close('all')
    ax = visualise.heatmap(persian_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# vis_miss() tests
# ---------------------------------------------------------------------------

def test_vis_miss_returns_axes(nan_df):
    """vis_miss() returns an Axes instance."""
    plt.close('all')
    ax = visualise.vis_miss(nan_df)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_title(nan_df):
    """vis_miss() sets the expected title."""
    plt.close('all')
    ax = visualise.vis_miss(nan_df)
    assert ax.get_title() == 'Missing Data Overview'


def test_vis_miss_sentinel(sample_df):
    """vis_miss() works with sentinel missing_values."""
    plt.close('all')
    ax = visualise.vis_miss(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_vis_miss_no_pct(nan_df):
    """vis_miss() does not raise when show_pct=False."""
    plt.close('all')
    ax = visualise.vis_miss(nan_df, show_pct=False)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_cluster(nan_df):
    """vis_miss() with cluster=True does not raise."""
    plt.close('all')
    ax = visualise.vis_miss(nan_df, cluster=True)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_persian(persian_df):
    """vis_miss() handles Persian column names and index labels."""
    plt.close('all')
    ax = visualise.vis_miss(persian_df)
    assert isinstance(ax, plt.Axes)
    # Title must still be in English per API contract.
    assert ax.get_title() == 'Missing Data Overview'


def test_vis_miss_all_missing(all_missing_df):
    """vis_miss() handles a fully-missing DataFrame without raising."""
    plt.close('all')
    ax = visualise.vis_miss(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_no_missing(no_missing_df):
    """vis_miss() handles a fully-observed DataFrame without raising."""
    plt.close('all')
    ax = visualise.vis_miss(no_missing_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_var_pct() tests
# ---------------------------------------------------------------------------

def test_miss_var_pct_returns_axes(nan_df):
    """miss_var_pct() returns an Axes instance."""
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_title(nan_df):
    """miss_var_pct() sets the expected title."""
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df)
    assert ax.get_title() == 'Missing Values per Variable (%)'


def test_miss_var_pct_sentinel(sample_df):
    """miss_var_pct() works with sentinel missing_values."""
    plt.close('all')
    ax = visualise.miss_var_pct(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_no_sort(nan_df):
    """miss_var_pct() with sort=False does not raise."""
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df, sort=False)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_values(nan_df):
    """miss_var_pct() bars should reflect correct percentages.

    X and Y each have 50 % missing; Z has 0 %.
    The function must produce bars whose lengths match these values.
    """
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df, sort=False)
    # barh containers: one patch per variable
    widths = [patch.get_width() for patch in ax.patches]
    assert any(abs(w - 50.0) < 0.1 for w in widths), (
        f"Expected a 50 % bar; got widths={widths}"
    )
    assert any(abs(w - 0.0) < 0.1 for w in widths), (
        f"Expected a 0 % bar; got widths={widths}"
    )


def test_miss_var_pct_persian(persian_df):
    """miss_var_pct() handles Persian column names without raising."""
    plt.close('all')
    ax = visualise.miss_var_pct(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_no_missing(no_missing_df):
    """miss_var_pct() handles a fully-observed DataFrame without raising."""
    plt.close('all')
    ax = visualise.miss_var_pct(no_missing_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_cluster() tests
# ---------------------------------------------------------------------------

def test_miss_cluster_returns_axes(nan_df):
    """miss_cluster() returns an Axes instance."""
    plt.close('all')
    ax = visualise.miss_cluster(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_title(nan_df):
    """miss_cluster() sets the expected title."""
    plt.close('all')
    ax = visualise.miss_cluster(nan_df)
    assert ax.get_title() == 'Clustered Missing Data Matrix'


def test_miss_cluster_sentinel(sample_df):
    """miss_cluster() works with sentinel missing_values."""
    plt.close('all')
    ax = visualise.miss_cluster(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_all_missing(all_missing_df):
    """miss_cluster() handles a fully-missing DataFrame without raising."""
    plt.close('all')
    ax = visualise.miss_cluster(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_no_missing(no_missing_df):
    """miss_cluster() handles a fully-observed DataFrame without raising."""
    plt.close('all')
    ax = visualise.miss_cluster(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_persian(persian_df):
    """miss_cluster() handles Persian column names without raising."""
    plt.close('all')
    ax = visualise.miss_cluster(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_method(nan_df):
    """miss_cluster() accepts alternative linkage methods."""
    plt.close('all')
    ax = visualise.miss_cluster(nan_df, method='complete')
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_which() tests
# ---------------------------------------------------------------------------

def test_miss_which_returns_axes(nan_df):
    """miss_which() returns an Axes instance."""
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_title(nan_df):
    """miss_which() sets the expected title."""
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    assert ax.get_title() == 'Which Variables Have Missing Data?'


def test_miss_which_sentinel(sample_df):
    """miss_which() works with sentinel missing_values."""
    plt.close('all')
    ax = visualise.miss_which(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_which_no_missing(no_missing_df):
    """miss_which() handles a fully-observed DataFrame without raising."""
    plt.close('all')
    ax = visualise.miss_which(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_all_missing(all_missing_df):
    """miss_which() handles a fully-missing DataFrame without raising."""
    plt.close('all')
    ax = visualise.miss_which(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_persian(persian_df):
    """miss_which() handles Persian column names without raising."""
    plt.close('all')
    ax = visualise.miss_which(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_content(nan_df):
    """miss_which() tile for fully-observed column must differ from missing column.

    In nan_df, Z has no missing values.  The heatmap data for Z must be
    0.0, while X and Y (which have missing values) must be 1.0.
    """
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    # The heatmap collection stores the underlying data matrix.
    data = ax.collections[0].get_array()
    # data is flattened row-major: columns are X, Y, Z in order.
    col_names = list(nan_df.columns)          # ['X', 'Y', 'Z']
    z_idx = col_names.index('Z')
    x_idx = col_names.index('X')
    assert data[x_idx] == pytest.approx(1.0), "X should be flagged as missing"
    assert data[z_idx] == pytest.approx(0.0), "Z should not be flagged as missing"
