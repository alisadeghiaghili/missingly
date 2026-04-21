import pandas as pd
import numpy as np
import pytest

from missingly import manipulation

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': [1, -99, 3, 4],
        'B': ['x', 'y', 'N/A', 'z'],
        'C': [10.0, 20.0, 30.0, -99.0]
    }
    return pd.DataFrame(data)

def test_replace_with_na(sample_df):
    """Test replacing values with a dictionary."""
    df_replaced = manipulation.replace_with_na(sample_df, replace={'A': -99, 'B': 'N/A', 'C': -99.0})
    assert pd.isnull(df_replaced.loc[1, 'A'])
    assert pd.isnull(df_replaced.loc[2, 'B'])
    assert pd.isnull(df_replaced.loc[3, 'C'])
    assert not pd.isnull(df_replaced.loc[0, 'A'])
    assert not pd.isnull(df_replaced.loc[1, 'B'])

def test_replace_with_na_callable(sample_df):
    """Test replacing values with a callable."""
    df_replaced = manipulation.replace_with_na(sample_df, replace={'A': lambda x: x < 0})
    assert pd.isnull(df_replaced.loc[1, 'A'])
    assert not pd.isnull(df_replaced.loc[0, 'A'])

def test_replace_with_na_all(sample_df):
    """Test replacing values based on a condition."""
    df_replaced = manipulation.replace_with_na_all(sample_df, condition=lambda x: x == -99 or x == 'N/A')
    assert pd.isnull(df_replaced.loc[1, 'A'])
    assert pd.isnull(df_replaced.loc[2, 'B'])
    assert pd.isnull(df_replaced.loc[3, 'C'])
    assert not pd.isnull(df_replaced.loc[0, 'A'])
    assert not pd.isnull(df_replaced.loc[1, 'B'])
