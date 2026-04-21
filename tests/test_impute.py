import pandas as pd
import numpy as np
import pytest

from missingly import impute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Numeric-only dataframe with missing values."""
    return pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [10.0, np.nan, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def mixed_df():
    """Mixed numeric + categorical dataframe with missing values."""
    return pd.DataFrame({
        'age':      [25.0, 30.0, np.nan, 45.0, 35.0],
        'income':   [50000.0, np.nan, 75000.0, 60000.0, 80000.0],
        'education': ['HS', 'College', None, 'Graduate', 'College'],
        'gender':    ['M', None, 'F', 'M', 'F'],
    })


@pytest.fixture
def cat_only_df():
    """Categorical-only dataframe with missing values."""
    return pd.DataFrame({
        'color':  ['red', 'blue', None, 'red', 'green'],
        'size':   ['S', None, 'M', 'L', 'S'],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_nulls(df: pd.DataFrame) -> bool:
    return not df.isnull().any().any()


def _cat_values_valid(original: pd.DataFrame, imputed: pd.DataFrame, col: str) -> bool:
    """All imputed values in a categorical column must be from the original vocabulary."""
    vocab = set(original[col].dropna().unique())
    return set(imputed[col].unique()).issubset(vocab)


# ---------------------------------------------------------------------------
# impute_mean
# ---------------------------------------------------------------------------

class TestImputeMean:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_mean(numeric_df)
        assert _no_nulls(result)
        assert result.loc[2, 'A'] == pytest.approx(np.mean([1, 2, 4, 5]))
        assert result.loc[1, 'B'] == pytest.approx(np.mean([10, 30, 40, 50]))

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_mean(mixed_df)
        assert _no_nulls(result)

    def test_mixed_numeric_values_correct(self, mixed_df):
        result = impute.impute_mean(mixed_df)
        assert result.loc[2, 'age'] == pytest.approx(np.mean([25, 30, 45, 35]))

    def test_mixed_cat_imputed_with_mode(self, mixed_df):
        result = impute.impute_mean(mixed_df)
        assert _cat_values_valid(mixed_df, result, 'education')
        assert _cat_values_valid(mixed_df, result, 'gender')


# ---------------------------------------------------------------------------
# impute_median
# ---------------------------------------------------------------------------

class TestImputeMedian:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_median(numeric_df)
        assert _no_nulls(result)
        assert result.loc[2, 'A'] == pytest.approx(np.median([1, 2, 4, 5]))

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_median(mixed_df)
        assert _no_nulls(result)

    def test_mixed_cat_imputed_with_mode(self, mixed_df):
        result = impute.impute_median(mixed_df)
        assert _cat_values_valid(mixed_df, result, 'education')


# ---------------------------------------------------------------------------
# impute_mode
# ---------------------------------------------------------------------------

class TestImputeMode:
    def test_numeric_no_nulls(self, numeric_df):
        df = pd.DataFrame({'A': [1, 2, 2, np.nan], 'B': [10, 20, 20, np.nan]})
        result = impute.impute_mode(df)
        assert _no_nulls(result)
        assert float(result.loc[3, 'A']) == pytest.approx(2.0)

    def test_cat_only_no_nulls(self, cat_only_df):
        result = impute.impute_mode(cat_only_df)
        assert _no_nulls(result)

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_mode(mixed_df)
        assert _no_nulls(result)


# ---------------------------------------------------------------------------
# impute_knn
# ---------------------------------------------------------------------------

class TestImputeKNN:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_knn(numeric_df, n_neighbors=2)
        assert _no_nulls(result)

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_knn(mixed_df, n_neighbors=2)
        assert _no_nulls(result)

    def test_mixed_cat_values_valid(self, mixed_df):
        result = impute.impute_knn(mixed_df, n_neighbors=2)
        assert _cat_values_valid(mixed_df, result, 'education')
        assert _cat_values_valid(mixed_df, result, 'gender')

    def test_cat_only_no_nulls(self, cat_only_df):
        result = impute.impute_knn(cat_only_df, n_neighbors=2)
        assert _no_nulls(result)


# ---------------------------------------------------------------------------
# impute_mice
# ---------------------------------------------------------------------------

class TestImputeMICE:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_mice(numeric_df)
        assert _no_nulls(result)

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_mice(mixed_df)
        assert _no_nulls(result)

    def test_mixed_cat_values_valid(self, mixed_df):
        result = impute.impute_mice(mixed_df)
        assert _cat_values_valid(mixed_df, result, 'education')
        assert _cat_values_valid(mixed_df, result, 'gender')


# ---------------------------------------------------------------------------
# impute_rf
# ---------------------------------------------------------------------------

class TestImputeRF:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_rf(numeric_df)
        assert _no_nulls(result)

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_rf(mixed_df)
        assert _no_nulls(result)

    def test_mixed_cat_values_valid(self, mixed_df):
        result = impute.impute_rf(mixed_df)
        assert _cat_values_valid(mixed_df, result, 'education')
        assert _cat_values_valid(mixed_df, result, 'gender')


# ---------------------------------------------------------------------------
# impute_gb
# ---------------------------------------------------------------------------

class TestImputeGB:
    def test_numeric_no_nulls(self, numeric_df):
        result = impute.impute_gb(numeric_df)
        assert _no_nulls(result)

    def test_mixed_no_nulls(self, mixed_df):
        result = impute.impute_gb(mixed_df)
        assert _no_nulls(result)

    def test_mixed_cat_values_valid(self, mixed_df):
        result = impute.impute_gb(mixed_df)
        assert _cat_values_valid(mixed_df, result, 'education')
        assert _cat_values_valid(mixed_df, result, 'gender')
