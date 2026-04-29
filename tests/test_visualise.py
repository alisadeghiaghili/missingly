"""Tests for missingly.visualise module.

Each test:
* closes all open figures before running to avoid state leakage;
* calls the public API through the top-level ``missingly`` namespace
  where the intent is to verify the import path, and through
  ``missingly.visualise`` for direct module tests;
* asserts on the returned Axes type and plot title.
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


@pytest.fixture
def sample_df():
    """Numeric dataframe with sentinel missing values (-99)."""
    return pd.DataFrame({
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0],
    })


# ---------------------------------------------------------------------------
# Top-level import smoke test
# ---------------------------------------------------------------------------

def test_dendrogram_importable_from_top_level():
    """dendrogram must be accessible via `from missingly import dendrogram`.

    This test catches the class of bug where a symbol is listed in
    ``__all__`` but never imported in ``__init__.py``.
    """
    assert callable(missingly.dendrogram), (
        "missingly.dendrogram is not callable — check __init__.py imports"
    )


# ---------------------------------------------------------------------------
# visualise tests
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
