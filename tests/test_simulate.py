"""Tests for the missing data simulation module.

Conventions
-----------
* We test the *contract* (return type, no mutation, column names, dtype
  preservation, fraction accuracy) rather than exact masked indices,
  which are seed-dependent.
* The MAR and MNAR tests verify that the chosen mechanism actually
  produces a *directional* bias: e.g. MNAR-upper should mask
  disproportionately more rows from the upper half of the column.
* Error-path tests cover every documented ValueError.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.simulate import (
    simulate_mcar,
    simulate_mar,
    simulate_mnar,
    simulate_mixed,
)
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def complete_df():
    """300-row complete DataFrame with numeric and string columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'age':    rng.integers(20, 65, 300).astype(float),
        'income': rng.normal(50_000, 10_000, 300),
        'score':  rng.normal(0, 1, 300),
        'city':   np.resize(['Paris', 'Lyon', 'Nice'], 300),
    })


@pytest.fixture
def numeric_df():
    """500-row purely numeric complete DataFrame."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        'a': rng.normal(0, 1, 500),
        'b': rng.normal(10, 2, 500),
        'c': rng.normal(-5, 3, 500),
    })


# ---------------------------------------------------------------------------
# simulate_mcar — contract
# ---------------------------------------------------------------------------

def test_mcar_returns_dataframe(complete_df):
    result = simulate_mcar(complete_df, frac=0.10, random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_mcar_does_not_mutate(complete_df):
    original = complete_df.copy(deep=True)
    simulate_mcar(complete_df, frac=0.10, random_state=0)
    pd.testing.assert_frame_equal(complete_df, original)


def test_mcar_columns_preserved(complete_df):
    result = simulate_mcar(complete_df, frac=0.10, random_state=0)
    assert list(result.columns) == list(complete_df.columns)


def test_mcar_index_preserved(complete_df):
    result = simulate_mcar(complete_df, frac=0.10, random_state=0)
    pd.testing.assert_index_equal(result.index, complete_df.index)


def test_mcar_approximate_frac(numeric_df):
    """Masked fraction should be within 5pp of requested frac."""
    result = simulate_mcar(numeric_df, frac=0.20, random_state=42)
    for col in numeric_df.columns:
        actual = result[col].isnull().mean()
        assert abs(actual - 0.20) < 0.05, f"{col}: frac={actual:.3f}"


def test_mcar_subset_columns(complete_df):
    """Only the specified columns should contain NaNs."""
    result = simulate_mcar(complete_df, frac=0.15, columns=['age', 'score'],
                           random_state=0)
    assert result['income'].isnull().sum() == 0
    assert result['city'].isnull().sum() == 0
    assert result['age'].isnull().sum() > 0
    assert result['score'].isnull().sum() > 0


def test_mcar_reproducible(complete_df):
    r1 = simulate_mcar(complete_df, frac=0.10, random_state=7)
    r2 = simulate_mcar(complete_df, frac=0.10, random_state=7)
    pd.testing.assert_frame_equal(r1, r2)


def test_mcar_different_seeds_differ(complete_df):
    r1 = simulate_mcar(complete_df, frac=0.20, random_state=1)
    r2 = simulate_mcar(complete_df, frac=0.20, random_state=2)
    assert not r1.equals(r2)


# ---------------------------------------------------------------------------
# simulate_mcar — error paths
# ---------------------------------------------------------------------------

def test_mcar_raises_on_existing_missing(complete_df):
    df_dirty = complete_df.copy()
    df_dirty.loc[0, 'age'] = np.nan
    with pytest.raises(ValueError, match="complete DataFrame"):
        simulate_mcar(df_dirty, frac=0.10)


def test_mcar_raises_invalid_frac(complete_df):
    with pytest.raises(ValueError, match="frac"):
        simulate_mcar(complete_df, frac=1.5)


def test_mcar_raises_unknown_column(complete_df):
    with pytest.raises(ValueError, match="not found"):
        simulate_mcar(complete_df, columns=['nonexistent'])


# ---------------------------------------------------------------------------
# simulate_mar — contract
# ---------------------------------------------------------------------------

def test_mar_returns_dataframe(numeric_df):
    result = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                          frac=0.10, random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_mar_only_target_col_has_missing(numeric_df):
    result = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                          frac=0.15, random_state=0)
    assert result['b'].isnull().sum() == 0
    assert result['c'].isnull().sum() == 0
    assert result['a'].isnull().sum() > 0


def test_mar_does_not_mutate(numeric_df):
    original = numeric_df.copy(deep=True)
    simulate_mar(numeric_df, target_col='a', predictor_col='b',
                 frac=0.10, random_state=0)
    pd.testing.assert_frame_equal(numeric_df, original)


def test_mar_upper_tail_bias(numeric_df):
    """Upper-tail MAR: masked rows should have above-median predictor values."""
    result = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                          frac=0.30, tail='upper', random_state=0)
    median_b = numeric_df['b'].median()
    masked_idx = result[result['a'].isnull()].index
    pct_above = (numeric_df.loc[masked_idx, 'b'] > median_b).mean()
    # With 30% masking and upper-tail weighting, > 60% should be above median
    assert pct_above > 0.60, f"Expected >60% above median, got {pct_above:.2%}"


def test_mar_lower_tail_bias(numeric_df):
    """Lower-tail MAR: masked rows should have below-median predictor values."""
    result = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                          frac=0.30, tail='lower', random_state=0)
    median_b = numeric_df['b'].median()
    masked_idx = result[result['a'].isnull()].index
    pct_below = (numeric_df.loc[masked_idx, 'b'] < median_b).mean()
    assert pct_below > 0.60, f"Expected >60% below median, got {pct_below:.2%}"


def test_mar_reproducible(numeric_df):
    r1 = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                      frac=0.10, random_state=5)
    r2 = simulate_mar(numeric_df, target_col='a', predictor_col='b',
                      frac=0.10, random_state=5)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# simulate_mar — error paths
# ---------------------------------------------------------------------------

def test_mar_raises_invalid_tail(numeric_df):
    with pytest.raises(ValueError, match="tail"):
        simulate_mar(numeric_df, target_col='a', predictor_col='b',
                     tail='middle')


def test_mar_raises_non_numeric_predictor(complete_df):
    with pytest.raises(ValueError, match="numeric"):
        simulate_mar(complete_df, target_col='age',
                     predictor_col='city', frac=0.10)


def test_mar_raises_existing_missing(complete_df):
    df_dirty = complete_df.copy()
    df_dirty.loc[0, 'age'] = np.nan
    with pytest.raises(ValueError, match="complete DataFrame"):
        simulate_mar(df_dirty, target_col='score', predictor_col='income')


# ---------------------------------------------------------------------------
# simulate_mnar — contract
# ---------------------------------------------------------------------------

def test_mnar_returns_dataframe(numeric_df):
    result = simulate_mnar(numeric_df, target_col='a', frac=0.10,
                           random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_mnar_only_target_col_has_missing(numeric_df):
    result = simulate_mnar(numeric_df, target_col='a', frac=0.15,
                           random_state=0)
    assert result['b'].isnull().sum() == 0
    assert result['a'].isnull().sum() > 0


def test_mnar_upper_tail_bias(numeric_df):
    """MNAR upper: masked values should be above the column median."""
    result = simulate_mnar(numeric_df, target_col='a', frac=0.30,
                           tail='upper', random_state=0)
    median_a = numeric_df['a'].median()
    masked_idx = result[result['a'].isnull()].index
    pct_above = (numeric_df.loc[masked_idx, 'a'] > median_a).mean()
    assert pct_above > 0.60


def test_mnar_lower_tail_bias(numeric_df):
    """MNAR lower: masked values should be below the column median."""
    result = simulate_mnar(numeric_df, target_col='a', frac=0.30,
                           tail='lower', random_state=0)
    median_a = numeric_df['a'].median()
    masked_idx = result[result['a'].isnull()].index
    pct_below = (numeric_df.loc[masked_idx, 'a'] < median_a).mean()
    assert pct_below > 0.60


def test_mnar_does_not_mutate(numeric_df):
    original = numeric_df.copy(deep=True)
    simulate_mnar(numeric_df, target_col='a', frac=0.10, random_state=0)
    pd.testing.assert_frame_equal(numeric_df, original)


# ---------------------------------------------------------------------------
# simulate_mnar — error paths
# ---------------------------------------------------------------------------

def test_mnar_raises_non_numeric_target(complete_df):
    with pytest.raises(ValueError, match="numeric"):
        simulate_mnar(complete_df, target_col='city', frac=0.10)


def test_mnar_raises_invalid_tail(numeric_df):
    with pytest.raises(ValueError, match="tail"):
        simulate_mnar(numeric_df, target_col='a', frac=0.10, tail='both')


# ---------------------------------------------------------------------------
# simulate_mixed — contract
# ---------------------------------------------------------------------------

def test_mixed_returns_dataframe(numeric_df):
    spec = [
        {'mechanism': 'MCAR', 'columns': ['c'], 'frac': 0.10},
        {'mechanism': 'MAR',  'target_col': 'a', 'predictor_col': 'b',
         'frac': 0.10},
        {'mechanism': 'MNAR', 'target_col': 'b', 'frac': 0.10},
    ]
    result = simulate_mixed(numeric_df, spec, random_state=0)
    assert isinstance(result, pd.DataFrame)


def test_mixed_all_target_cols_have_missing(numeric_df):
    spec = [
        {'mechanism': 'MCAR', 'columns': ['c'], 'frac': 0.15},
        {'mechanism': 'MAR',  'target_col': 'a', 'predictor_col': 'b',
         'frac': 0.15},
        {'mechanism': 'MNAR', 'target_col': 'b', 'frac': 0.15},
    ]
    result = simulate_mixed(numeric_df, spec, random_state=0)
    assert result['a'].isnull().sum() > 0
    assert result['b'].isnull().sum() > 0
    assert result['c'].isnull().sum() > 0


def test_mixed_does_not_mutate(numeric_df):
    original = numeric_df.copy(deep=True)
    spec = [{'mechanism': 'MCAR', 'frac': 0.10}]
    simulate_mixed(numeric_df, spec, random_state=0)
    pd.testing.assert_frame_equal(numeric_df, original)


def test_mixed_reproducible(numeric_df):
    spec = [
        {'mechanism': 'MCAR', 'frac': 0.10},
        {'mechanism': 'MNAR', 'target_col': 'a', 'frac': 0.10},
    ]
    r1 = simulate_mixed(numeric_df, spec, random_state=99)
    r2 = simulate_mixed(numeric_df, spec, random_state=99)
    pd.testing.assert_frame_equal(r1, r2)


def test_mixed_raises_unknown_mechanism(numeric_df):
    spec = [{'mechanism': 'MAGIC', 'frac': 0.10}]
    with pytest.raises(ValueError, match="unknown mechanism"):
        simulate_mixed(numeric_df, spec)


def test_mixed_raises_existing_missing(numeric_df):
    df_dirty = numeric_df.copy()
    df_dirty.loc[0, 'a'] = np.nan
    spec = [{'mechanism': 'MCAR', 'frac': 0.10}]
    with pytest.raises(ValueError, match="complete DataFrame"):
        simulate_mixed(df_dirty, spec)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_simulate_functions_importable_from_top_level():
    assert callable(missingly.simulate_mcar)
    assert callable(missingly.simulate_mar)
    assert callable(missingly.simulate_mnar)
    assert callable(missingly.simulate_mixed)
