"""Unit tests for missingly.exceptions."""

import pytest

from missingly.exceptions import (
    ConfigurationError,
    ImputationError,
    InsufficientDataError,
    InvalidStrategyError,
    MissingColumnError,
    MissinglyError,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

def test_all_exceptions_inherit_from_missingly_error():
    assert issubclass(ImputationError, MissinglyError)
    assert issubclass(InsufficientDataError, MissinglyError)
    assert issubclass(InvalidStrategyError, MissinglyError)
    assert issubclass(MissingColumnError, MissinglyError)
    assert issubclass(ConfigurationError, MissinglyError)


# ---------------------------------------------------------------------------
# ImputationError
# ---------------------------------------------------------------------------

def test_imputation_error_message():
    original = ValueError("singular matrix")
    err = ImputationError(column="age", strategy="rf", original=original)
    assert "age" in str(err)
    assert "rf" in str(err)
    assert "singular matrix" in str(err)


def test_imputation_error_attributes():
    original = RuntimeError("boom")
    err = ImputationError(column="x", strategy="gb", original=original)
    assert err.column == "x"
    assert err.strategy == "gb"
    assert err.original is original


def test_imputation_error_is_catchable_as_missingly_error():
    with pytest.raises(MissinglyError):
        raise ImputationError(column="x", strategy="knn", original=ValueError())


# ---------------------------------------------------------------------------
# InsufficientDataError
# ---------------------------------------------------------------------------

def test_insufficient_data_error_message():
    err = InsufficientDataError(column="salary", n_observed=0, n_required=1)
    assert "salary" in str(err)
    assert "0" in str(err)
    assert "1" in str(err)


def test_insufficient_data_error_attributes():
    err = InsufficientDataError(column="col", n_observed=2, n_required=5)
    assert err.column == "col"
    assert err.n_observed == 2
    assert err.n_required == 5


# ---------------------------------------------------------------------------
# InvalidStrategyError
# ---------------------------------------------------------------------------

def test_invalid_strategy_error_message():
    err = InvalidStrategyError(param="strategy", got="avg", allowed=["mean", "median"])
    assert "strategy" in str(err)
    assert "avg" in str(err)
    assert "mean" in str(err)


def test_invalid_strategy_error_with_example():
    err = InvalidStrategyError(
        param="metric", got="l2", allowed=["euclidean", "mixed"], example="euclidean"
    )
    assert "euclidean" in str(err)
    assert "Try" in str(err)


def test_invalid_strategy_error_attributes():
    err = InvalidStrategyError(param="p", got="bad", allowed=["a", "b"])
    assert err.param == "p"
    assert err.got == "bad"
    assert err.allowed == ["a", "b"]


# ---------------------------------------------------------------------------
# MissingColumnError
# ---------------------------------------------------------------------------

def test_missing_column_error_message():
    err = MissingColumnError(columns=["age"], available=["name", "salary"])
    assert "age" in str(err)
    assert "name" in str(err)
    assert "salary" in str(err)


def test_missing_column_error_multiple_columns():
    err = MissingColumnError(columns=["a", "b"], available=["c"])
    assert "a" in str(err)
    assert "b" in str(err)


def test_missing_column_error_attributes():
    err = MissingColumnError(columns=["x"], available=["y", "z"])
    assert err.columns == ["x"]
    assert err.available == ["y", "z"]


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------

def test_configuration_error_message():
    err = ConfigurationError(
        param="large_df_threshold", got=-1, reason="must be a positive integer"
    )
    assert "large_df_threshold" in str(err)
    assert "-1" in str(err)
    assert "positive integer" in str(err)


def test_configuration_error_attributes():
    err = ConfigurationError(param="p", got=0, reason="reason")
    assert err.param == "p"
    assert err.got == 0
    assert err.reason == "reason"
