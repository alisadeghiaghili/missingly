import pandas as pd
import numpy as np
import pytest

from missingly import impute

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50]
    }
    return pd.DataFrame(data)

def test_impute_mean(sample_df):
    """Test mean imputation."""
    df_imputed = impute.impute_mean(sample_df)
    assert not df_imputed.isnull().any().any()
    assert df_imputed.loc[2, 'A'] == np.mean([1, 2, 4, 5])
    assert df_imputed.loc[1, 'B'] == np.mean([10, 30, 40, 50])

def test_impute_median(sample_df):
    """Test median imputation."""
    df_imputed = impute.impute_median(sample_df)
    assert not df_imputed.isnull().any().any()
    assert df_imputed.loc[2, 'A'] == np.median([1, 2, 4, 5])
    assert df_imputed.loc[1, 'B'] == np.median([10, 30, 40, 50])

def test_impute_mode(sample_df):
    """Test mode imputation."""
    df = pd.DataFrame({
        'A': [1, 2, 2, np.nan],
        'B': [10, 20, 20, np.nan]
    })
    df_imputed = impute.impute_mode(df)
    assert not df_imputed.isnull().any().any()
    assert df_imputed.loc[3, 'A'] == 2
    assert df_imputed.loc[3, 'B'] == 20

def test_impute_knn(sample_df):
    """Test KNN imputation."""
    df_imputed = impute.impute_knn(sample_df, n_neighbors=2)
    assert not df_imputed.isnull().any().any()
    # The exact imputed value is hard to predict, so we just check it's not null.
    assert pd.notnull(df_imputed.loc[2, 'A'])
    assert pd.notnull(df_imputed.loc[1, 'B'])

def test_impute_mice(sample_df):
    """Test MICE imputation."""
    df_imputed = impute.impute_mice(sample_df)
    assert not df_imputed.isnull().any().any()

def test_impute_rf(sample_df):
    """Test Random Forest imputation."""
    df_imputed = impute.impute_rf(sample_df)
    assert not df_imputed.isnull().any().any()

def test_impute_gb(sample_df):
    """Test Gradient Boosting imputation."""
    df_imputed = impute.impute_gb(sample_df)
    assert not df_imputed.isnull().any().any()
