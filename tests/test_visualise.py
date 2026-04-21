import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt

from missingly import visualise
from missingly import impute

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0]
    }
    return pd.DataFrame(data)

def test_matrix(sample_df):
    """Test that matrix runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.matrix(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Data Matrix'

def test_bar(sample_df):
    """Test that bar runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.bar(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Column'

def test_upset(sample_df):
    """Test that upset runs without error."""
    plt.close('all')
    # Upsetplot returns a dict of axes
    plot = visualise.upset(sample_df, missing_values=[-99])
    assert isinstance(plot, dict)

def test_scatter_miss(sample_df):
    """Test that scatter_miss runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.scatter_miss(sample_df, x='A', y='B', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Scatter Plot of A vs B with Missing Values'

def test_miss_case(sample_df):
    """Test that miss_case runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.miss_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values per Case'

def test_vis_impute_dist(sample_df):
    """Test that vis_impute_dist runs without error and returns an axes object."""
    plt.close('all')
    imputed_df = impute.impute_mean(sample_df)
    ax = visualise.vis_impute_dist(sample_df, imputed_df, 'A')
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Distribution of Original vs. Imputed Data for A'

def test_vis_miss_fct(sample_df):
    """Test that vis_miss_fct runs without error and returns an axes object."""
    plt.close('all')
    df = sample_df.copy()
    df['Fct'] = ['a', 'b', 'a', 'b']
    ax = visualise.vis_miss_fct(df, 'Fct', missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values by Fct'

def test_vis_miss_cumsum_var(sample_df):
    """Test that vis_miss_cumsum_var runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.vis_miss_cumsum_var(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Variable'

def test_vis_miss_cumsum_case(sample_df):
    """Test that vis_miss_cumsum_case runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.vis_miss_cumsum_case(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Cumulative Sum of Missing Values per Case'

def test_vis_miss_span(sample_df):
    """Test that vis_miss_span runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.vis_miss_span(sample_df, 'A', 2, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Missing Values in Spans of 2 for A'

def test_vis_parallel_coords(sample_df):
    """Test that vis_parallel_coords runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.vis_parallel_coords(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Parallel Coordinates Plot of Missingness'

# Add this test function to your test_visualise.py file

def test_dendrogram(sample_df):
    """Test that dendrogram runs without error and returns an axes object."""
    plt.close('all')
    ax = visualise.dendrogram(sample_df, missing_values=[-99])
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Dendrogram of Variables by Missing Data Patterns'
