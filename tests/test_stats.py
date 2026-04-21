import pandas as pd
import numpy as np
import pytest

from missingly import stats

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0]
    }
    return pd.DataFrame(data)

def test_mcar_test(sample_df):
    """Test mcar_test with custom missing values."""
    result = stats.mcar_test(sample_df, missing_values=[-99])
    assert 'p_value' in result
    assert result['p_value'] > 0.05

def test_mar_mnar_test(sample_df):
    """Test mar_mnar_test with custom missing values."""
    Y = np.random.randint(0, 2, 4)
    result = stats.mar_mnar_test(sample_df, Y, missing_values=[-99])
    assert isinstance(result, list)
