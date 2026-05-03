"""Tests for missingly imputation functions.

Covers correctness, mutation safety, categorical support, large-data
warnings, and the new RF/GB classifier-for-categoricals behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    impute_mice,
    impute_rf,
    impute_gb,
    _LARGE_DF_ROW_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Small numeric-only DataFrame with missing values."""
    return pd.DataFrame({
        "age":    [25.0, np.nan, 35.0, 40.0, 30.0],
        "income": [50_000.0, 60_000.0, np.nan, 80_000.0, 70_000.0],
        "score":  [85.0, 90.0, 78.0, np.nan, 88.0],
    })


@pytest.fixture
def mixed_df():
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "age":    [25.0, np.nan, 35.0, 40.0, 30.0, 28.0],
        "city":   ["Paris", "London", np.nan, "Berlin", "Paris", "London"],
        "score":  [85.0, 90.0, 78.0, np.nan, 88.0, 92.0],
        "grade":  ["A", "B", "A", "C", np.nan, "B"],
    })


@pytest.fixture
def cat_only_df():
    """DataFrame with only categorical columns."""
    return pd.DataFrame({
        "color": ["red", "blue", np.nan, "green", "red"],
        "size":  ["S", np.nan, "M", "L", "S"],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_missing(df: pd.DataFrame) -> None:
    """Assert that a DataFrame contains no missing values."""
    assert df.isnull().sum().sum() == 0, (
        f"Expected no missing values; found {df.isnull().sum().sum()}"
    )


def _assert_not_mutated(original: pd.DataFrame, result: pd.DataFrame) -> None:
    """Assert that the result is a distinct object from the original."""
    assert result is not original


# ---------------------------------------------------------------------------
# Simple imputers — numeric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_no_missing(fn, numeric_df):
    """Simple imputers produce a fully-observed numeric DataFrame."""
    result = fn(numeric_df)
    _assert_no_missing(result)
    _assert_not_mutated(numeric_df, result)


@pytest.mark.parametrize("fn", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_mixed(fn, mixed_df):
    """Simple imputers handle mixed numeric/categorical DataFrames."""
    result = fn(mixed_df)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


@pytest.mark.parametrize("fn", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_cat_only(fn, cat_only_df):
    """Simple imputers handle categorical-only DataFrames."""
    result = fn(cat_only_df)
    _assert_no_missing(result)


# ---------------------------------------------------------------------------
# Mean imputer — value checks
# ---------------------------------------------------------------------------

def test_impute_mean_correct_value(numeric_df):
    """Mean imputer fills with column mean."""
    result = impute_mean(numeric_df)
    expected_age = numeric_df["age"].mean()
    assert abs(result.loc[1, "age"] - expected_age) < 1e-6


# ---------------------------------------------------------------------------
# KNN imputer
# ---------------------------------------------------------------------------

def test_impute_knn_no_missing(numeric_df):
    """KNN imputer produces a fully-observed DataFrame."""
    result = impute_knn(numeric_df, n_neighbors=2)
    _assert_no_missing(result)
    _assert_not_mutated(numeric_df, result)


def test_impute_knn_mixed(mixed_df):
    """KNN imputer handles mixed DataFrames."""
    result = impute_knn(mixed_df, n_neighbors=2)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


def test_impute_knn_large_warning():
    """KNN imputer emits UserWarning for large DataFrames."""
    large_df = pd.DataFrame(
        {"a": np.random.default_rng(0).random(_LARGE_DF_ROW_THRESHOLD + 1)}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        impute_knn(large_df)
    assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# MICE imputer
# ---------------------------------------------------------------------------

def test_impute_mice_no_missing(numeric_df):
    """MICE imputer produces a fully-observed DataFrame."""
    result = impute_mice(numeric_df, max_iter=3)
    _assert_no_missing(result)
    _assert_not_mutated(numeric_df, result)


def test_impute_mice_mixed(mixed_df):
    """MICE imputer handles mixed DataFrames."""
    result = impute_mice(mixed_df, max_iter=3)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


def test_impute_mice_large_warning():
    """MICE imputer emits UserWarning for large DataFrames."""
    large_df = pd.DataFrame(
        {"a": np.random.default_rng(0).random(_LARGE_DF_ROW_THRESHOLD + 1)}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        impute_mice(large_df)
    assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# RF imputer — regressor for numeric, classifier for categorical
# ---------------------------------------------------------------------------

def test_impute_rf_no_missing(numeric_df):
    """RF imputer produces a fully-observed numeric DataFrame."""
    result = impute_rf(numeric_df)
    _assert_no_missing(result)
    _assert_not_mutated(numeric_df, result)


def test_impute_rf_mixed_no_missing(mixed_df):
    """RF imputer produces a fully-observed mixed DataFrame."""
    result = impute_rf(mixed_df)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


def test_impute_rf_categorical_values_valid(mixed_df):
    """RF imputer produces valid category values for categorical columns."""
    result = impute_rf(mixed_df)
    valid_cities = set(mixed_df["city"].dropna())
    imputed_cities = set(result["city"])
    # All imputed city values must come from the original categories
    assert imputed_cities.issubset(valid_cities), (
        f"RF imputed invalid city values: {imputed_cities - valid_cities}"
    )


def test_impute_rf_large_warning():
    """RF imputer emits UserWarning for large DataFrames."""
    large_df = pd.DataFrame(
        {"a": np.random.default_rng(0).random(_LARGE_DF_ROW_THRESHOLD + 1)}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        impute_rf(large_df)
    assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# GB imputer — regressor for numeric, classifier for categorical
# ---------------------------------------------------------------------------

def test_impute_gb_no_missing(numeric_df):
    """GB imputer produces a fully-observed numeric DataFrame."""
    result = impute_gb(numeric_df)
    _assert_no_missing(result)
    _assert_not_mutated(numeric_df, result)


def test_impute_gb_mixed_no_missing(mixed_df):
    """GB imputer produces a fully-observed mixed DataFrame."""
    result = impute_gb(mixed_df)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


def test_impute_gb_categorical_values_valid(mixed_df):
    """GB imputer produces valid category values for categorical columns."""
    result = impute_gb(mixed_df)
    valid_grades = set(mixed_df["grade"].dropna())
    imputed_grades = set(result["grade"])
    assert imputed_grades.issubset(valid_grades), (
        f"GB imputed invalid grade values: {imputed_grades - valid_grades}"
    )


def test_impute_gb_large_warning():
    """GB imputer emits UserWarning for large DataFrames."""
    large_df = pd.DataFrame(
        {"a": np.random.default_rng(0).random(_LARGE_DF_ROW_THRESHOLD + 1)}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        impute_gb(large_df)
    assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# None normalisation
# ---------------------------------------------------------------------------

def test_none_normalised_to_nan():
    """Python None in object columns is treated as missing by all imputers."""
    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": ["x", None, "z"]})
    for fn in [impute_mean, impute_median, impute_mode, impute_knn,
               impute_mice, impute_rf, impute_gb]:
        result = fn(df)
        _assert_no_missing(result)
