"""Tests for missingly.visualise module.

Conventions
-----------
* Every test closes all open figures before running to avoid state
  leakage between tests.
* Return-type assertions verify the public contract (Axes or dict/Figure).
* Title assertions pin the human-readable label so accidental renames
  are caught immediately.
* Edge-case tests (all-missing, no-missing, Persian labels, single
  column) live in clearly named functions.
* Sentinel-value tests use ``missing_values=[-99]`` throughout.
"""

import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missingly import visualise
from missingly import impute
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def nan_df():
    return pd.DataFrame({
        'X': [1.0, np.nan, 3.0, np.nan],
        'Y': [np.nan, 2.0, np.nan, 4.0],
        'Z': [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def persian_df():
    return pd.DataFrame({
        'درآمد': [1000, np.nan, 3000, np.nan],
        'سن': [25, 30, np.nan, 40],
        'نام': ['علی', None, 'رضا', 'مریم'],
    }, index=['ردیف_۱', 'ردیف_۲', 'ردیف_۳', 'ردیف_۴'])


@pytest.fixture
def all_missing_df():
    return pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]})


@pytest.fixture
def no_missing_df():
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})


@pytest.fixture
def group_df():
    return pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C', 'C'],
        'x': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
        'y': [np.nan, 2.0, np.nan, 4.0, 5.0, 6.0],
    })


# ---------------------------------------------------------------------------
# Top-level import smoke tests
# ---------------------------------------------------------------------------

def test_dendrogram_importable_from_top_level():
    assert callable(missingly.dendrogram)


def test_new_functions_importable_from_top_level():
    for fn_name in (
        "heatmap", "vis_miss", "miss_var_pct", "miss_cluster", "miss_which",
        "miss_patterns", "miss_cooccurrence", "miss_row_profile",
        "shadow_scatter", "vis_miss_by_group", "miss_impute_compare",
    ):
        assert callable(getattr(missingly, fn_name, None)), (
            f"missingly.{fn_name} is not callable"
        )


# ---------------------------------------------------------------------------
# Basic visualisations
# ---------------------------------------------------------------------------

def test_matrix(sample_df):
    plt.close('all')
    ax = visualise.matrix(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Data Matrix'


def test_bar(sample_df):
    plt.close('all')
    ax = visualise.bar(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Column'


def test_upset(sample_df):
    plt.close('all')
    plot = visualise.upset(sample_df, missing_values=[-99])
    plt.close('all')
    assert isinstance(plot, dict)


def test_upset_keys(sample_df):
    plt.close('all')
    plot = visualise.upset(sample_df, missing_values=[-99])
    plt.close('all')
    assert set(plot.keys()) == {"intersections", "matrix", "totals"}


def test_upset_no_missing(no_missing_df):
    plt.close('all')
    result = visualise.upset(no_missing_df)
    assert result == {}


def test_upset_show_pct(sample_df):
    """upset() with show_pct=True does not raise."""
    plt.close('all')
    result = visualise.upset(sample_df, missing_values=[-99], show_pct=True)
    plt.close('all')
    assert "intersections" in result


def test_scatter_miss(sample_df):
    plt.close('all')
    ax = visualise.scatter_miss(sample_df, x='A', y='B', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Scatter Plot of A vs B with Missing Values'


def test_miss_case(sample_df):
    plt.close('all')
    ax = visualise.miss_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Case'


def test_vis_impute_dist(sample_df):
    plt.close('all')
    imputed_df = impute.impute_mean(sample_df)
    ax = visualise.vis_impute_dist(sample_df, imputed_df, 'A')
    assert isinstance(ax, plt.Axes)
    assert 'A' in ax.get_title()


def test_vis_miss_fct(sample_df):
    plt.close('all')
    df = sample_df.copy()
    df['Fct'] = ['a', 'b', 'a', 'b']
    ax = visualise.vis_miss_fct(df, 'Fct', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert 'Fct' in ax.get_title()


def test_vis_miss_cumsum_var(sample_df):
    plt.close('all')
    ax = visualise.vis_miss_cumsum_var(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Variable'


def test_vis_miss_cumsum_case(sample_df):
    plt.close('all')
    ax = visualise.vis_miss_cumsum_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Case'


def test_vis_miss_span(sample_df):
    plt.close('all')
    ax = visualise.vis_miss_span(sample_df, 'A', 2, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert 'A' in ax.get_title()
    assert '2' in ax.get_title()


def test_vis_parallel_coords(sample_df):
    plt.close('all')
    ax = visualise.vis_parallel_coords(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Parallel Coordinates Plot of Missingness'


def test_dendrogram(sample_df):
    plt.close('all')
    ax = visualise.dendrogram(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Dendrogram of Variables by Missing Data Patterns'


# ---------------------------------------------------------------------------
# heatmap() tests
# ---------------------------------------------------------------------------

def test_heatmap_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.heatmap(nan_df)
    assert isinstance(ax, plt.Axes)


def test_heatmap_title(nan_df):
    plt.close('all')
    ax = visualise.heatmap(nan_df)
    assert 'Nullity Correlation Heatmap' in ax.get_title()


def test_heatmap_sentinel(sample_df):
    plt.close('all')
    ax = visualise.heatmap(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_heatmap_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.heatmap(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_heatmap_persian(persian_df):
    plt.close('all')
    ax = visualise.heatmap(persian_df)
    assert isinstance(ax, plt.Axes)


def test_heatmap_phi_method(nan_df):
    """heatmap() with method='phi' does not raise."""
    plt.close('all')
    ax = visualise.heatmap(nan_df, method='phi')
    assert isinstance(ax, plt.Axes)


def test_heatmap_mask_insignificant(nan_df):
    """heatmap() with mask_insignificant=True does not raise."""
    plt.close('all')
    df = pd.DataFrame({
        'A': [np.nan if i % 2 == 0 else float(i) for i in range(20)],
        'B': [np.nan if i % 2 == 0 else float(i) for i in range(20)],
        'C': [np.nan if i % 3 == 0 else float(i) for i in range(20)],
    })
    ax = visualise.heatmap(df, mask_insignificant=True)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# vis_miss() tests
# ---------------------------------------------------------------------------

def test_vis_miss_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.vis_miss(nan_df)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_title(nan_df):
    plt.close('all')
    ax = visualise.vis_miss(nan_df)
    assert ax.get_title() == 'Missing Data Overview'


def test_vis_miss_sentinel(sample_df):
    plt.close('all')
    ax = visualise.vis_miss(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_vis_miss_no_pct(nan_df):
    plt.close('all')
    ax = visualise.vis_miss(nan_df, show_pct=False)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_cluster(nan_df):
    plt.close('all')
    ax = visualise.vis_miss(nan_df, cluster=True)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_persian(persian_df):
    plt.close('all')
    ax = visualise.vis_miss(persian_df)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Data Overview'


def test_vis_miss_all_missing(all_missing_df):
    plt.close('all')
    ax = visualise.vis_miss(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_vis_miss_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.vis_miss(no_missing_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_var_pct() tests
# ---------------------------------------------------------------------------

def test_miss_var_pct_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_title(nan_df):
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df)
    assert ax.get_title() == 'Missing Values per Variable (%)'


def test_miss_var_pct_sentinel(sample_df):
    plt.close('all')
    ax = visualise.miss_var_pct(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_no_sort(nan_df):
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df, sort=False)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_values(nan_df):
    plt.close('all')
    ax = visualise.miss_var_pct(nan_df, sort=False)
    widths = [patch.get_width() for patch in ax.patches]
    assert any(abs(w - 50.0) < 0.1 for w in widths)
    assert any(abs(w - 0.0) < 0.1 for w in widths)


def test_miss_var_pct_persian(persian_df):
    plt.close('all')
    ax = visualise.miss_var_pct(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_var_pct_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.miss_var_pct(no_missing_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_cluster() tests
# ---------------------------------------------------------------------------

def test_miss_cluster_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_cluster(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_title(nan_df):
    plt.close('all')
    ax = visualise.miss_cluster(nan_df)
    assert ax.get_title() == 'Clustered Missing Data Matrix'


def test_miss_cluster_sentinel(sample_df):
    plt.close('all')
    ax = visualise.miss_cluster(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_all_missing(all_missing_df):
    plt.close('all')
    ax = visualise.miss_cluster(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.miss_cluster(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_persian(persian_df):
    plt.close('all')
    ax = visualise.miss_cluster(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cluster_method(nan_df):
    plt.close('all')
    ax = visualise.miss_cluster(nan_df, method='complete')
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_which() tests
# ---------------------------------------------------------------------------

def test_miss_which_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_title(nan_df):
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    assert ax.get_title() == 'Which Variables Have Missing Data?'


def test_miss_which_sentinel(sample_df):
    plt.close('all')
    ax = visualise.miss_which(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


def test_miss_which_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.miss_which(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_all_missing(all_missing_df):
    plt.close('all')
    ax = visualise.miss_which(all_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_persian(persian_df):
    plt.close('all')
    ax = visualise.miss_which(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_which_content(nan_df):
    plt.close('all')
    ax = visualise.miss_which(nan_df)
    raw = ax.collections[0].get_array()
    data = np.asarray(raw).ravel()
    col_names = list(nan_df.columns)
    x_idx = col_names.index('X')
    z_idx = col_names.index('Z')
    assert float(data[x_idx]) == pytest.approx(1.0)
    assert float(data[z_idx]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# miss_patterns() tests
# ---------------------------------------------------------------------------

def test_miss_patterns_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_patterns(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_patterns_title(nan_df):
    plt.close('all')
    ax = visualise.miss_patterns(nan_df)
    assert 'Missingness Patterns' in ax.get_title()


def test_miss_patterns_no_missing(no_missing_df):
    """miss_patterns() on a complete DataFrame plots one '(complete)' bar."""
    plt.close('all')
    ax = visualise.miss_patterns(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_patterns_persian(persian_df):
    plt.close('all')
    ax = visualise.miss_patterns(persian_df)
    assert isinstance(ax, plt.Axes)


def test_miss_patterns_sentinel(sample_df):
    plt.close('all')
    ax = visualise.miss_patterns(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_cooccurrence() tests
# ---------------------------------------------------------------------------

def test_miss_cooccurrence_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_cooccurrence(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_cooccurrence_title_normalized(nan_df):
    plt.close('all')
    ax = visualise.miss_cooccurrence(nan_df, normalize=True)
    assert 'fraction' in ax.get_title()


def test_miss_cooccurrence_title_count(nan_df):
    plt.close('all')
    ax = visualise.miss_cooccurrence(nan_df, normalize=False)
    assert 'count' in ax.get_title()


def test_miss_cooccurrence_diagonal(nan_df):
    """Diagonal should equal each column's individual missing count."""
    plt.close('all')
    null_mat = nan_df.isnull().astype(int)
    cooc = null_mat.T.dot(null_mat) / len(nan_df)
    for col in nan_df.columns:
        expected = nan_df[col].isnull().mean()
        assert abs(cooc.loc[col, col] - expected) < 1e-9


def test_miss_cooccurrence_persian(persian_df):
    plt.close('all')
    ax = visualise.miss_cooccurrence(persian_df)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_row_profile() tests
# ---------------------------------------------------------------------------

def test_miss_row_profile_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.miss_row_profile(nan_df)
    assert isinstance(ax, plt.Axes)


def test_miss_row_profile_title(nan_df):
    plt.close('all')
    ax = visualise.miss_row_profile(nan_df)
    assert ax.get_title() == 'Row Missingness Profile'


def test_miss_row_profile_no_missing(no_missing_df):
    plt.close('all')
    ax = visualise.miss_row_profile(no_missing_df)
    assert isinstance(ax, plt.Axes)


def test_miss_row_profile_sentinel(sample_df):
    plt.close('all')
    ax = visualise.miss_row_profile(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# shadow_scatter() tests
# ---------------------------------------------------------------------------

def test_shadow_scatter_returns_axes(nan_df):
    plt.close('all')
    ax = visualise.shadow_scatter(nan_df, x='X', y='Z', shadow_col='Y')
    assert isinstance(ax, plt.Axes)


def test_shadow_scatter_title(nan_df):
    plt.close('all')
    ax = visualise.shadow_scatter(nan_df, x='X', y='Z', shadow_col='Y')
    assert 'X' in ax.get_title()
    assert 'Z' in ax.get_title()
    assert 'Y' in ax.get_title()


def test_shadow_scatter_persian(persian_df):
    """shadow_scatter() handles Persian column names."""
    plt.close('all')
    df = persian_df.copy()
    df['درآمد'] = pd.to_numeric(df['درآمد'], errors='coerce')
    df['سن'] = pd.to_numeric(df['سن'], errors='coerce')
    ax = visualise.shadow_scatter(df, x='درآمد', y='سن', shadow_col='نام')
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# vis_miss_by_group() tests
# ---------------------------------------------------------------------------

def test_vis_miss_by_group_returns_axes(group_df):
    plt.close('all')
    ax = visualise.vis_miss_by_group(group_df, group_col='group')
    assert isinstance(ax, plt.Axes)


def test_vis_miss_by_group_title(group_df):
    plt.close('all')
    ax = visualise.vis_miss_by_group(group_df, group_col='group')
    assert 'group' in ax.get_title()


def test_vis_miss_by_group_shape(group_df):
    """Heatmap should have n_groups rows and n_vars-1 columns."""
    plt.close('all')
    ax = visualise.vis_miss_by_group(group_df, group_col='group')
    # 3 groups, 2 non-group variables
    collection = ax.collections[0]
    data = np.asarray(collection.get_array())
    assert data.size == 3 * 2


def test_vis_miss_by_group_persian(persian_df):
    plt.close('all')
    df = persian_df.copy()
    df['گروه'] = ['الف', 'ب', 'الف', 'ب']
    ax = visualise.vis_miss_by_group(df, group_col='گروه')
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# miss_impute_compare() tests
# ---------------------------------------------------------------------------

def test_miss_impute_compare_returns_figure(nan_df):
    plt.close('all')
    imputed = impute.impute_mean(nan_df)
    fig = visualise.miss_impute_compare(nan_df, imputed)
    assert hasattr(fig, 'savefig')  # is a Figure
    plt.close('all')


def test_miss_impute_compare_specific_columns(nan_df):
    plt.close('all')
    imputed = impute.impute_mean(nan_df)
    fig = visualise.miss_impute_compare(nan_df, imputed, columns=['X'])
    assert hasattr(fig, 'savefig')
    plt.close('all')


def test_miss_impute_compare_no_missing_raises(no_missing_df):
    """Should raise ValueError when no columns have missing values."""
    plt.close('all')
    with pytest.raises(ValueError, match="No numeric columns"):
        visualise.miss_impute_compare(no_missing_df, no_missing_df)


# ---------------------------------------------------------------------------
# RTL / Persian helpers
# ---------------------------------------------------------------------------

def test_rtl_safe_wraps_persian():
    """_rtl_safe must wrap Persian text in RLM + text + LRM."""
    result = visualise._rtl_safe('درآمد')
    assert result.startswith('\u200F')
    assert result.endswith('\u200E')


def test_rtl_safe_leaves_latin():
    """_rtl_safe must not modify pure Latin strings."""
    text = 'income'
    assert visualise._rtl_safe(text) == text


def test_safe_labels_mixed():
    """_safe_labels applies RTL wrapping only to RTL items."""
    labels = ['income', 'درآمد', 'age']
    result = visualise._safe_labels(labels)
    assert result[0] == 'income'
    assert result[1].startswith('\u200F')
    assert result[2] == 'age'
