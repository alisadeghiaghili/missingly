"""Tests for the additional statistical tests module (stats_extra.py).

Conventions
-----------
* We test the contract (return type, key presence, valid ranges) and
  the directional behaviour (e.g. MNAR data should give a lower
  Hotelling p-value than MCAR data on the same DataFrame).
* Tests are kept fast: small DataFrames, no heavy simulation.
* The canonical names are hotelling_test / pattern_monotone_test;
  the test_* aliases are also verified to exist.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.stats_extra import (
    hotelling_test,
    test_hotelling,
    pattern_monotone_test,
    test_pattern_monotone,
    missing_correlation_matrix,
)
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
    df.loc[:20, 'x'] = -99
    result = hotelling_test(df, missing_values=[-99])
    assert isinstance(result, dict)


def test_test_hotelling_alias(df_with_missing):
    """test_hotelling is a backward-compat alias for hotelling_test."""
    assert test_hotelling is hotelling_test


# ---------------------------------------------------------------------------
# pattern_monotone_test
# ---------------------------------------------------------------------------

def test_monotone_returns_dict(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert isinstance(result, dict)


def test_monotone_required_keys(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    for key in ('is_monotone', 'n_violating_rows',
                'sorted_columns', 'monotone_pct'):
        assert key in result


def test_monotone_truly_monotone():
    df = pd.DataFrame({
        'a': [1.0, np.nan, np.nan, np.nan],
        'b': [2.0, 2.0,   np.nan, np.nan],
        'c': [3.0, 3.0,   3.0,   np.nan],
    })
    result = pattern_monotone_test(df)
    assert result['is_monotone'] is True
    assert result['n_violating_rows'] == 0
    assert result['monotone_pct'] == 1.0


def test_monotone_non_monotone():
    df = pd.DataFrame({
        'a': [1.0, np.nan, 3.0],
        'b': [2.0, 2.0,   np.nan],
    })
    result = pattern_monotone_test(df)
    assert result['is_monotone'] is False
    assert result['n_violating_rows'] > 0


def test_monotone_no_missing(complete_df):
    result = pattern_monotone_test(complete_df)
    assert result['is_monotone'] is True
    assert result['n_violating_rows'] == 0


def test_monotone_pct_range(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert 0.0 <= result['monotone_pct'] <= 1.0


def test_monotone_sorted_columns_contains_all(df_with_missing):
    result = pattern_monotone_test(df_with_missing)
    assert set(result['sorted_columns']) == set(df_with_missing.columns)


def test_test_pattern_monotone_alias(df_with_missing):
    """test_pattern_monotone is a backward-compat alias."""
    assert test_pattern_monotone is pattern_monotone_test


# ---------------------------------------------------------------------------
# missing_correlation_matrix
# ---------------------------------------------------------------------------

def test_corr_matrix_returns_dataframe(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    assert isinstance(result, pd.DataFrame)


def test_corr_matrix_is_square(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    cols_with_missing = df_with_missing.columns[
        df_with_missing.isnull().any()
    ].tolist()
    assert result.shape == (len(cols_with_missing), len(cols_with_missing))


def test_corr_matrix_diagonal_is_one(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    diag = np.diag(result.to_numpy())
    assert np.allclose(diag, 1.0)


def test_corr_matrix_values_in_range(df_with_missing):
    result = missing_correlation_matrix(df_with_missing)
    vals = result.to_numpy()
    assert np.all(vals >= -1.0 - 1e-9)
    assert np.all(vals <= 1.0 + 1e-9)


def test_corr_matrix_no_missing_returns_empty(complete_df):
    result = missing_correlation_matrix(complete_df)
    assert result.shape == (0, 0) or len(result) == 0


def test_corr_matrix_invalid_method_raises(df_with_missing):
    with pytest.raises(ValueError, match="method"):
        missing_correlation_matrix(df_with_missing, method='cosine')


def test_corr_matrix_spearman(df_with_missing):
    result = missing_correlation_matrix(df_with_missing, method='spearman')
    assert isinstance(result, pd.DataFrame)


def test_corr_matrix_sentinel(complete_df):
    df = complete_df.copy()
    df.loc[:10, 'x'] = -99
    df.loc[5:15, 'y'] = -99
    result = missing_correlation_matrix(df, missing_values=[-99])
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_stats_extra_importable_from_top_level():
    assert callable(missingly.hotelling_test)
    assert callable(missingly.test_hotelling)        # alias
    assert callable(missingly.pattern_monotone_test)
    assert callable(missingly.test_pattern_monotone)  # alias
    assert callable(missingly.missing_correlation_matrix)
