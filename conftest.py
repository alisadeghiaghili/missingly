"""Root conftest.py — shared fixtures for the missingly test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures — reusable across all test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def df_no_missing():
    """Small complete DataFrame with numeric and string columns."""
    return pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0, 45.0],
        "city": ["Berlin", "Munich", "Hamburg", "Berlin", "Munich"],
        "score": [85.0, 90.0, 78.0, 92.0, 88.0],
    })


@pytest.fixture
df_simple():
    """Small numeric DataFrame with some missing values."""
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0],
        "b": [np.nan, 2.0, np.nan, 4.0],
    })


@pytest.fixture
def df_all_missing():
    """DataFrame where every value is missing."""
    return pd.DataFrame({
        "a": [np.nan, np.nan, np.nan],
        "b": [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def df_single_row():
    """Single-row DataFrame."""
    return pd.DataFrame({"a": [1.0], "b": ["x"]})


@pytest.fixture
def df_single_col():
    """Single-column DataFrame."""
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})


@pytest.fixture
def df_mixed_dtypes():
    """DataFrame with numeric, string, and categorical columns."""
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 40.0, 30.0],
        "city": ["Berlin", "Munich", np.nan, "Berlin", "Hamburg"],
        "grade": pd.Categorical(["A", "B", "A", "C", np.nan], categories=["A", "B", "C"]),
        "score": [85.0, 90.0, 78.0, np.nan, 88.0],
    })


@pytest.fixture
def df_large():
    """Large DataFrame (1000 rows) for performance testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 1000),
        "b": rng.normal(10, 2, 1000),
        "c": rng.choice(["x", "y", "z"], 1000),
    })


@pytest.fixture
def df_high_cardinality_cat():
    """DataFrame with a high-cardinality categorical column."""
    rng = np.random.default_rng(0)
    categories = [f"cat_{i}" for i in range(100)]
    return pd.DataFrame({
        "id": range(500),
        "category": rng.choice(categories, 500),
        "value": rng.normal(0, 1, 500),
    })


@pytest.fixture
def df_datetime():
    """DataFrame with a DatetimeIndex for time-series tests."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "value": rng.normal(0, 1, 100),
        "count": rng.integers(0, 100, 100),
    }, index=dates)
