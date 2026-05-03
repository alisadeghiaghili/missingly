"""Tests for missingly.stats — mcar_test, mar_mnar_test, diagnose_missing.

Conventions
-----------
* All fixtures use fixed random seeds so results are deterministic.
* diagnose_missing tests assert on 'mechanism' (enum-like string) rather
  than the free-text 'recommendation' field so tests aren't brittle to
  wording changes.
* We do NOT assert exact p-values because those depend on numerical EM
  convergence; instead we assert on the derived 'mechanism' field which
  is computed from the p-value relative to the significance threshold.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.stats import mcar_test, mar_mnar_test, diagnose_missing
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df_with_missing():
    """Small numeric DataFrame with clear missing structure."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'A': rng.normal(0, 1, 30),
        'B': rng.normal(5, 2, 30),
        'C': rng.normal(-1, 0.5, 30),
    })
    # Introduce ~20% missing at random
    for col in df.columns:
        idx = rng.choice(30, size=6, replace=False)
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture
def small_numeric_df():
    """Minimal 4-row numeric DataFrame for edge-case tests."""
    return pd.DataFrame({
        'X': [1.0, np.nan, 3.0, 4.0],
        'Y': [np.nan, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def no_missing_df():
    """Complete DataFrame — no missing values."""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [5.0, 6.0, 7.0, 8.0],
    })


@pytest.fixture
def single_col_df():
    """Single-column DataFrame — Little's test cannot run."""
    return pd.DataFrame({'A': [1.0, np.nan, 3.0, 4.0]})


@pytest.fixture
def high_miss_df():
    """DataFrame with >40% missingness in one column."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        'A': rng.normal(0, 1, 20),
        'B': rng.normal(0, 1, 20),
    })
    df.loc[:10, 'A'] = np.nan  # 55% missing in A
    return df


# ---------------------------------------------------------------------------
# mcar_test
# ---------------------------------------------------------------------------

def test_mcar_test_returns_dict(numeric_df_with_missing):
    """mcar_test must return a dict with the required keys."""
    result = mcar_test(numeric_df_with_missing)
    assert isinstance(result, dict)
    for key in ('chi_square', 'df', 'p_value', 'missing_patterns', 'amount_missing'):
        assert key in result, f"Missing key: {key}"


def test_mcar_test_p_value_range(numeric_df_with_missing):
    """p_value must be in [0, 1] or NaN."""
    result = mcar_test(numeric_df_with_missing)
    pv = result['p_value']
    if not np.isnan(pv):
        assert 0.0 <= pv <= 1.0


def test_mcar_test_chi_square_nonneg(numeric_df_with_missing):
    """chi_square statistic must be non-negative."""
    result = mcar_test(numeric_df_with_missing)
    assert result['chi_square'] >= 0.0


def test_mcar_test_amount_missing_shape(numeric_df_with_missing):
    """amount_missing must be a (2, n_cols) DataFrame."""
    result = mcar_test(numeric_df_with_missing)
    am = result['amount_missing']
    assert isinstance(am, pd.DataFrame)
    assert am.shape == (2, numeric_df_with_missing.shape[1])


def test_mcar_test_sentinel(numeric_df_with_missing):
    """mcar_test handles sentinel missing_values without raising."""
    df = numeric_df_with_missing.fillna(-99)
    result = mcar_test(df, missing_values=[-99])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# mar_mnar_test
# ---------------------------------------------------------------------------

def test_mar_mnar_test_returns_list(numeric_df_with_missing):
    """mar_mnar_test must return a list."""
    Y = np.random.default_rng(0).integers(0, 2, len(numeric_df_with_missing))
    result = mar_mnar_test(numeric_df_with_missing, Y)
    assert isinstance(result, list)


def test_mar_mnar_test_tuple_structure(numeric_df_with_missing):
    """Each element must be a 3-tuple (feature_name, LRT, p_value)."""
    Y = np.random.default_rng(0).integers(0, 2, len(numeric_df_with_missing))
    result = mar_mnar_test(numeric_df_with_missing, Y)
    for item in result:
        assert len(item) == 3
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)
        assert 0.0 <= item[2] <= 1.0


# ---------------------------------------------------------------------------
# diagnose_missing — return-type contract
# ---------------------------------------------------------------------------

def test_diagnose_returns_dict(numeric_df_with_missing):
    """diagnose_missing must return a dict."""
    result = diagnose_missing(numeric_df_with_missing)
    assert isinstance(result, dict)


def test_diagnose_required_keys(numeric_df_with_missing):
    """All required keys must be present in the result."""
    result = diagnose_missing(numeric_df_with_missing)
    for key in (
        'mechanism', 'recommendation', 'strategy_hint',
        'high_missingness_cols', 'max_nullity_corr',
        'chi_square', 'p_value',
    ):
        assert key in result, f"Missing key: {key}"


def test_diagnose_mechanism_valid_values(numeric_df_with_missing):
    """mechanism must be one of the four defined strings."""
    result = diagnose_missing(numeric_df_with_missing)
    assert result['mechanism'] in (
        'MCAR', 'MAR', 'possible_MNAR', 'insufficient_data'
    )


def test_diagnose_recommendation_nonempty(numeric_df_with_missing):
    """recommendation must be a non-empty string."""
    result = diagnose_missing(numeric_df_with_missing)
    assert isinstance(result['recommendation'], str)
    assert len(result['recommendation']) > 20


def test_diagnose_strategy_hint_nonempty(numeric_df_with_missing):
    """strategy_hint must be a non-empty string."""
    result = diagnose_missing(numeric_df_with_missing)
    assert isinstance(result['strategy_hint'], str)
    assert len(result['strategy_hint']) > 0


# ---------------------------------------------------------------------------
# diagnose_missing — edge cases
# ---------------------------------------------------------------------------

def test_diagnose_insufficient_data_single_col(single_col_df):
    """Single numeric column → insufficient_data (can't run Little's test)."""
    result = diagnose_missing(single_col_df)
    assert result['mechanism'] == 'insufficient_data'


def test_diagnose_insufficient_data_no_missing(no_missing_df):
    """No missing numeric values → insufficient_data (no missing cols to test)."""
    result = diagnose_missing(no_missing_df)
    assert result['mechanism'] == 'insufficient_data'


def test_diagnose_high_missingness_flagged(high_miss_df):
    """Columns with >40% missing must appear in high_missingness_cols."""
    result = diagnose_missing(high_miss_df)
    assert 'A' in result['high_missingness_cols']


def test_diagnose_no_high_missingness(small_numeric_df):
    """Columns with <=40% missing must NOT appear in high_missingness_cols."""
    result = diagnose_missing(small_numeric_df)
    # small_numeric_df has 25% missing per column — below the 40% threshold
    assert result['high_missingness_cols'] == []


def test_diagnose_max_nullity_corr_none_single_col(single_col_df):
    """max_nullity_corr must be None when there is only one numeric column."""
    result = diagnose_missing(single_col_df)
    assert result['max_nullity_corr'] is None


def test_diagnose_max_nullity_corr_nonneg(numeric_df_with_missing):
    """max_nullity_corr must be >= 0 when computable."""
    result = diagnose_missing(numeric_df_with_missing)
    if result['max_nullity_corr'] is not None:
        assert result['max_nullity_corr'] >= 0.0


def test_diagnose_sentinel(numeric_df_with_missing):
    """diagnose_missing respects missing_values sentinel parameter."""
    df = numeric_df_with_missing.fillna(-99)
    result = diagnose_missing(df, missing_values=[-99])
    assert isinstance(result, dict)
    assert result['mechanism'] in (
        'MCAR', 'MAR', 'possible_MNAR', 'insufficient_data'
    )


def test_diagnose_importable_from_top_level():
    """diagnose_missing must be importable from the missingly top-level package."""
    assert callable(missingly.diagnose_missing)
