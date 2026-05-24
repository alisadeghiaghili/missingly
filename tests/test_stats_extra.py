"""Tests for the additional statistical tests module (stats_extra.py)."""
import numpy as np
import pandas as pd
import pytest

from missingly.stats_extra import (
    hotelling_test,
    pattern_monotone_test,
    missing_correlation_matrix,
)
from missingly import test_hotelling, test_pattern_monotone
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_missing():
    """200-row DataFrame; column 'a' is MNAR-masked (upper tail)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'a': rng.normal(0, 1, 200),
        'b': rng.normal(5, 2, 200),
        'c': rng.normal(-1, 1, 200),
    })
    top30 = df['a'].nlargest(30).index
    df.loc[top30, 'a'] = np.nan
    return df


@pytest.fixture
def complete_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        'x': rng.normal(size=100),
        'y': rng.normal(size=100),
    })


# ---------------------------------------------------------------------------
# hotelling_test
# ---------------------------------------------------------------------------

def test_hotelling_returns_dict(df_with_missing):
    result = hotelling_test(df_with_missing)
    assert isinstance(result, dict)


def test_hotelling_required_keys(df_with_missing):
    result = hotelling_test(df_with_missing)
    for key in ('t2', 'f_statistic', 'df1', 'df2', 'p_value',
                'n_complete', 'n_incomplete', 'sufficient_data'):
        assert key in result, f"Missing key: {key}"


def test_hotelling_p_value_range(df_with_missing):
    result = hotelling_test(df_with_missing)
    if result['p_value'] is not None:
        assert 0.0 <= result['p_value'] <= 1.0


def test_hotelling_n_counts_correct(df_with_missing):
    result = hotelling_test(df_with_missing)
    assert result['n_complete'] + result['n_incomplete'] == len(df_with_missing)


def test_hotelling_t2_nonneg(df_with_missing):
    result = hotelling_test(df_with_missing)
    if result['t2'] is not None:
        assert result['t2'] >= 0


def test_hotelling_insufficient_data_no_missing(complete_df):
    result = hotelling_test(complete_df)
    assert result['sufficient_data'] is False
    assert result['p_value'] is None


def test_hotelling_raises_too_few_columns():
    df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="2 numeric columns"):
        hotelling_test(df)


def test_hotelling_sentinel(complete_df):
    df = complete_df.copy()
    df.iloc[0, 0] = -9999
    result = hotelling_test(df, missing_values=[-9999])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# pattern_monotone_test
# ---------------------------------------------------------------------------

def test_pattern_monotone_returns_dict(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert isinstance(result, dict)


def test_pattern_monotone_required_keys(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    for key in ('is_monotone', 'n_violating_rows', 'sorted_columns', 'monotone_pct'):
        assert key in result, f"Missing key: {key}"


def test_pattern_monotone_is_bool(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert isinstance(result['is_monotone'], bool)


def test_pattern_monotone_pct_range(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert 0.0 <= result['monotone_pct'] <= 1.0


def test_pattern_monotone_complete_data_is_monotone(complete_df):
    result = pattern_monotone_test(complete_df)
    assert result['is_monotone'] is True
    assert result['n_violating_rows'] == 0


def test_pattern_monotone_sentinel(complete_df):
    df = complete_df.copy()
    df.iloc[:5, 0] = -9999
    result = pattern_monotone_test(df, missing_values=[-9999])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# missing_correlation_matrix
# ---------------------------------------------------------------------------

def test_missing_corr_returns_dataframe(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    assert isinstance(result, pd.DataFrame)


def test_missing_corr_square(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    assert result.shape[0] == result.shape[1]


def test_missing_corr_diagonal_ones(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    if result.shape[0] > 0:
        diag = np.diag(result.values)
        assert np.allclose(diag, 1.0, atol=1e-9)


def test_missing_corr_empty_on_complete(complete_df):
    result = missing_correlation_matrix(complete_df)
    assert result.empty


def test_missing_corr_method_spearman(df_with_missing):
    result = missing_correlation_matrix(df_with_missing, method='spearman')
    assert isinstance(result, pd.DataFrame)


def test_missing_corr_invalid_method(df_with_missing):
    with pytest.raises(ValueError, match="method"):
        missing_correlation_matrix(df_with_missing, method='invalid')


# ---------------------------------------------------------------------------
# Alias tests (test_hotelling / test_pattern_monotone in missingly.__init__)
# ---------------------------------------------------------------------------

def test_alias_hotelling_is_callable():
    assert callable(test_hotelling)


def test_alias_pattern_monotone_is_callable():
    assert callable(test_pattern_monotone)


def test_alias_hotelling_returns_dict(df_with_missing):
    result = test_hotelling(df_with_missing)
    assert isinstance(result, dict)


def test_alias_pattern_monotone_returns_dict(df_with_missing):
    result = test_pattern_monotone(df_with_missing)
    assert isinstance(result, dict)


def test_aliases_are_same_function():
    """The aliases should resolve to the same underlying callable."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert test_hotelling.__wrapped__ is hotelling_test.__wrapped__
        assert test_pattern_monotone.__wrapped__ is pattern_monotone_test.__wrapped__
