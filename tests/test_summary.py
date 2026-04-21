import pandas as pd
import numpy as np
import pytest

from missingly import summary

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': [1, 2, -99, 4],
        'B': [-99, 'x', 'y', 'z'],
        'C': [10.0, 20.0, 30.0, 40.0],
        'D': [-99, -99, -99, -99]
    }
    return pd.DataFrame(data)

def test_n_miss(sample_df):
    """Test the n_miss function."""
    assert summary.n_miss(sample_df, missing_values=[-99]) == 6

def test_pct_miss(sample_df):
    """Test the pct_miss function."""
    assert summary.pct_miss(sample_df, missing_values=[-99]) == (6 / 16) * 100

def test_n_complete(sample_df):
    """Test the n_complete function."""
    assert summary.n_complete(sample_df, missing_values=[-99]) == 10

def test_pct_complete(sample_df):
    """Test the pct_complete function."""
    assert summary.pct_complete(sample_df, missing_values=[-99]) == (10 / 16) * 100

def test_miss_var_summary(sample_df):
    """Test the miss_var_summary function."""
    summary_df = summary.miss_var_summary(sample_df, missing_values=[-99])
    
    # Check the columns
    assert all(col in summary_df.columns for col in ['variable', 'n_miss', 'pct_miss'])
    
    # Check the values
    assert summary_df.loc[summary_df['variable'] == 'D', 'n_miss'].iloc[0] == 4
    assert summary_df.loc[summary_df['variable'] == 'A', 'n_miss'].iloc[0] == 1
    assert summary_df.loc[summary_df['variable'] == 'C', 'n_miss'].iloc[0] == 0

def test_miss_case_summary(sample_df):
    """Test the miss_case_summary function."""
    summary_df = summary.miss_case_summary(sample_df, missing_values=[-99])
    
    # Check the columns
    assert all(col in summary_df.columns for col in ['case', 'n_miss', 'pct_miss'])
    
    # Expected n_miss and pct_miss
    expected_n_miss = {0: 2, 1: 1, 2: 2, 3: 1}

    # Check the values for each row
    for i in range(4):
        assert summary_df.loc[summary_df['case'] == i, 'n_miss'].iloc[0] == expected_n_miss[i]
