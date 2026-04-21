import pandas as pd
import numpy as np
import pytest

from missingly import compare

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    }
    return pd.DataFrame(data)

def test_compare_imputations(sample_df):
    """Test that compare_imputations runs without error and returns a dataframe."""
    results = compare.compare_imputations(sample_df)
    
    assert isinstance(results, pd.DataFrame)
    assert 'RMSE' in results.columns
    assert len(results) > 0
