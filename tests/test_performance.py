"""Tests for the performance utilities module.

Conventions
-----------
* Tests avoid timing assertions (CI machines vary wildly in speed).
* We test the *contract*: return types, shapes, dtype changes, mutation
  safety, and error handling.
* String dtype is normalised via pandas api rather than comparing raw
  dtype strings, so tests pass on both Python <= 3.13 (object dtype)
  and Python 3.14+ (str / StringDtype).
"""

import numpy as np
import pandas as pd
import pytest

from missingly.performance import chunk_apply, memory_usage_mb, optimize_dtypes
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_large():
    """10 000-row mixed-dtype DataFrame for chunking tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'a': rng.normal(size=10_000),
        'b': rng.integers(0, 100, size=10_000, dtype='int64'),
        'c': np.resize(['Paris', 'Lyon', 'Nice'], 10_000),
    })


@pytest.fixture
def df_typed():
    """Small DataFrame for dtype optimisation tests."""
    return pd.DataFrame({
        'small_int': np.array([1, 2, 3, 4, 5], dtype='int64'),
        'big_float': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64'),
        'category':  ['a', 'b', 'a', 'b', 'a'],
        'high_card': ['x1', 'x2', 'x3', 'x4', 'x5'],
    })


# ---------------------------------------------------------------------------
# memory_usage_mb
# ---------------------------------------------------------------------------

def test_memory_usage_returns_dataframe(df_large):
    result = memory_usage_mb(df_large)
    assert isinstance(result, pd.DataFrame)


def test_memory_usage_has_required_columns(df_large):
    result = memory_usage_mb(df_large)
    assert 'dtype' in result.columns
    assert 'memory_mb' in result.columns


def test_memory_usage_has_total_row(df_large):
    result = memory_usage_mb(df_large)
    assert '__total__' in result.index


def test_memory_usage_total_equals_sum(df_large):
    result = memory_usage_mb(df_large)
    col_sum = result.drop(index='__total__')['memory_mb'].sum()
    total = result.loc['__total__', 'memory_mb']
    assert abs(col_sum - total) < 1e-9


def test_memory_usage_nonneg(df_large):
    result = memory_usage_mb(df_large)
    assert (result['memory_mb'] >= 0).all()


def test_memory_usage_does_not_mutate(df_large):
    original = df_large.copy(deep=True)
    memory_usage_mb(df_large)
    pd.testing.assert_frame_equal(df_large, original)


# ---------------------------------------------------------------------------
# optimize_dtypes
# ---------------------------------------------------------------------------

def test_optimize_returns_dataframe(df_typed):
    result = optimize_dtypes(df_typed)
    assert isinstance(result, pd.DataFrame)


def test_optimize_does_not_mutate(df_typed):
    original = df_typed.copy(deep=True)
    optimize_dtypes(df_typed)
    pd.testing.assert_frame_equal(df_typed, original)


def test_optimize_int_downcast():
    df = pd.DataFrame({'x': np.array([1, 2, 3], dtype='int64')})
    result = optimize_dtypes(df)
    assert result['x'].dtype == np.int8


def test_optimize_float_downcast():
    df = pd.DataFrame({'x': np.array([1.0, 2.0, 3.0], dtype='float64')})
    result = optimize_dtypes(df)
    assert result['x'].dtype == np.float32


def test_optimize_categorical_conversion():
    """Low-cardinality string/object column should become Categorical."""
    df = pd.DataFrame({'city': ['Paris', 'Lyon', 'Paris', 'Nice', 'Lyon']})
    result = optimize_dtypes(df, categorical_threshold=0.80)
    assert isinstance(result['city'].dtype, pd.CategoricalDtype)


def test_optimize_high_cardinality_not_converted():
    """100%-unique column must NOT be converted to Categorical."""
    df = pd.DataFrame({'id': [f'id_{i}' for i in range(100)]})
    result = optimize_dtypes(df, categorical_threshold=0.50)
    assert not isinstance(result['id'].dtype, pd.CategoricalDtype)


def test_optimize_categorical_disabled(df_typed):
    """When categorical_threshold=None, string columns must stay as-is."""
    result = optimize_dtypes(df_typed, categorical_threshold=None)
    assert not isinstance(result['category'].dtype, pd.CategoricalDtype)


def test_optimize_preserves_values(df_typed):
    result = optimize_dtypes(df_typed)
    pd.testing.assert_series_equal(
        result['small_int'].astype('int64'),
        df_typed['small_int'],
        check_names=False,
    )


def test_optimize_reduces_memory(df_typed):
    original_mb = df_typed.memory_usage(deep=True).sum()
    optimised_mb = optimize_dtypes(df_typed).memory_usage(deep=True).sum()
    assert optimised_mb < original_mb


# ---------------------------------------------------------------------------
# chunk_apply
# ---------------------------------------------------------------------------

def test_chunk_apply_returns_dataframe(df_large):
    result = chunk_apply(df_large, lambda x: x, chunk_size=1000)
    assert isinstance(result, pd.DataFrame)


def test_chunk_apply_same_shape(df_large):
    result = chunk_apply(df_large, lambda x: x, chunk_size=1000)
    assert result.shape == df_large.shape


def test_chunk_apply_same_values(df_large):
    result = chunk_apply(df_large, lambda x: x, chunk_size=1000)
    expected = df_large.reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


def test_chunk_apply_does_not_mutate(df_large):
    original = df_large.copy(deep=True)
    chunk_apply(df_large, lambda x: x, chunk_size=500)
    pd.testing.assert_frame_equal(df_large, original)


def test_chunk_apply_with_imputation():
    """chunk_apply works with impute_mean as a realistic use case."""
    from missingly.impute import impute_mean
    rng = np.random.default_rng(42)
    df = pd.DataFrame({'a': rng.normal(size=1000), 'b': rng.normal(size=1000)})
    df.loc[rng.choice(1000, 100, replace=False), 'a'] = np.nan

    result = chunk_apply(df, impute_mean, chunk_size=200)
    assert result['a'].isnull().sum() == 0
    assert result.shape == df.shape


def test_chunk_apply_single_chunk(df_large):
    """chunk_size > len(df) should behave like a direct call."""
    result = chunk_apply(df_large, lambda x: x, chunk_size=100_000)
    assert result.shape == df_large.shape


def test_chunk_apply_empty_df():
    df = pd.DataFrame({'a': [], 'b': []})
    result = chunk_apply(df, lambda x: x, chunk_size=100)
    assert result.shape == (0, 2)


def test_chunk_apply_raises_invalid_chunk_size(df_large):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_apply(df_large, lambda x: x, chunk_size=0)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_performance_functions_importable():
    assert callable(missingly.chunk_apply)
    assert callable(missingly.memory_usage_mb)
    assert callable(missingly.optimize_dtypes)
