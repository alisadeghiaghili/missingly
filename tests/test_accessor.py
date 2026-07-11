"""Tests for MissinglyAccessor.impute() and typed impute_* wrappers.

Tests follow the project convention: pytest, no Hypothesis, reuse inline fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import missingly  # noqa: F401 — registers df.miss accessor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_numeric():
    return pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]}
    )


@pytest.fixture()
def df_mixed():
    return pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0],
            "cat": ["x", None, "x", "y"],
        }
    )


@pytest.fixture()
def df_no_missing():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


@pytest.fixture()
def df_all_missing_col():
    return pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})


@pytest.fixture()
def df_categorical():
    return pd.DataFrame(
        {
            "city": pd.Categorical(
                ["Berlin", None, "Berlin", "Paris"],
                categories=["Berlin", "Paris"],
            ),
            "score": [1.0, 2.0, np.nan, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# 1. accessor exists after normal import
# ---------------------------------------------------------------------------

def test_impute_method_exists():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    assert hasattr(df.miss, "impute")


# ---------------------------------------------------------------------------
# 2. default call returns new DataFrame; source unchanged
# ---------------------------------------------------------------------------

def test_impute_default_returns_new_df(df_numeric):
    original_values = df_numeric.copy()
    result = df_numeric.miss.impute(method="mean")
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(df_numeric, original_values)  # source unchanged


def test_impute_default_no_missing_values(df_numeric):
    result = df_numeric.miss.impute(method="mean")
    assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 3. inplace=True mutates source and returns None
# ---------------------------------------------------------------------------

def test_impute_inplace_returns_none(df_numeric):
    result = df_numeric.miss.impute(method="mean", inplace=True)
    assert result is None


def test_impute_inplace_mutates_source(df_numeric):
    df_numeric.miss.impute(method="mean", inplace=True)
    assert df_numeric.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 4. Numeric imputation strategies
# ---------------------------------------------------------------------------

def test_impute_mean_numeric(df_numeric):
    result = df_numeric.miss.impute(method="mean")
    # mean of [1, 3, 4] == 2.666...
    assert abs(result.loc[1, "a"] - (1.0 + 3.0 + 4.0) / 3) < 1e-6


def test_impute_median_numeric(df_numeric):
    result = df_numeric.miss.impute(method="median")
    # median of [1, 3, 4] == 3.0
    assert result.loc[1, "a"] == pytest.approx(3.0)


def test_impute_mode_numeric(df_numeric):
    result = df_numeric.miss.impute(method="mode")
    assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 5. Categorical imputation and category preservation
# ---------------------------------------------------------------------------

def test_impute_mean_categorical_fallback_to_mode(df_mixed):
    result = df_mixed.miss.impute(method="mean")
    # cat column should be filled (mode = 'x')
    assert result["cat"].isna().sum() == 0
    assert result.loc[1, "cat"] == "x"


def test_impute_mode_categorical(df_mixed):
    result = df_mixed.miss.impute(method="mode")
    assert result["cat"].isna().sum() == 0


# ---------------------------------------------------------------------------
# 6. Selected-column behavior
# ---------------------------------------------------------------------------

def test_impute_selected_columns_only(df_numeric):
    original_b = df_numeric["b"].copy()
    result = df_numeric.miss.impute(method="mean", columns=["a"])
    # column a should be imputed
    assert result["a"].isna().sum() == 0
    # column b should be UNCHANGED
    pd.testing.assert_series_equal(result["b"], original_b)


def test_impute_preserves_column_order(df_numeric):
    result = df_numeric.miss.impute(method="mean")
    assert list(result.columns) == list(df_numeric.columns)


def test_impute_preserves_index(df_numeric):
    df_idx = df_numeric.copy()
    df_idx.index = [10, 20, 30, 40]
    result = df_idx.miss.impute(method="mean")
    assert list(result.index) == [10, 20, 30, 40]


# ---------------------------------------------------------------------------
# 7. Unknown column errors
# ---------------------------------------------------------------------------

def test_impute_unknown_columns_raises(df_numeric):
    with pytest.raises(Exception):  # MissingColumnError is a subclass of Exception
        df_numeric.miss.impute(method="mean", columns=["nonexistent"])


# ---------------------------------------------------------------------------
# 8. Unknown method errors
# ---------------------------------------------------------------------------

def test_impute_unknown_method_raises(df_numeric):
    with pytest.raises(ValueError, match="Unknown imputation method"):
        df_numeric.miss.impute(method="foobar")


# ---------------------------------------------------------------------------
# 9. Constant strategy
# ---------------------------------------------------------------------------

def test_impute_constant_numeric(df_numeric):
    result = df_numeric.miss.impute(method="constant", fill_value=-1.0)
    assert result.isna().sum().sum() == 0
    assert result.loc[1, "a"] == pytest.approx(-1.0)


def test_impute_constant_missing_fill_value_raises(df_numeric):
    with pytest.raises(ValueError, match="fill_value"):
        df_numeric.miss.impute(method="constant")


# ---------------------------------------------------------------------------
# 10. No missing values — passthrough
# ---------------------------------------------------------------------------

def test_impute_no_missing_passthrough(df_no_missing):
    result = df_no_missing.miss.impute(method="mean")
    pd.testing.assert_frame_equal(result, df_no_missing)


# ---------------------------------------------------------------------------
# 11. Mixed dtypes
# ---------------------------------------------------------------------------

def test_impute_mixed_dtypes(df_mixed):
    result = df_mixed.miss.impute(method="mean")
    assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 12. Parity with MissinglyImputer
# ---------------------------------------------------------------------------

def test_impute_parity_with_transformer(df_numeric):
    from missingly.transformer import MissinglyImputer
    accessor_result = df_numeric.miss.impute(method="mean")
    imp = MissinglyImputer(strategy="mean")
    transformer_result = imp.fit_transform(df_numeric)
    pd.testing.assert_frame_equal(
        accessor_result.reset_index(drop=True),
        transformer_result.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 13. Typed impute_* wrappers smoke tests
# ---------------------------------------------------------------------------

def test_typed_impute_mean_wrapper(df_numeric):
    result = df_numeric.miss.impute_mean()
    assert result.isna().sum().sum() == 0


def test_typed_impute_median_wrapper(df_numeric):
    result = df_numeric.miss.impute_median()
    assert result.isna().sum().sum() == 0


def test_typed_impute_mode_wrapper(df_mixed):
    result = df_mixed.miss.impute_mode()
    assert result.isna().sum().sum() == 0


def test_typed_impute_knn_wrapper(df_numeric):
    result = df_numeric.miss.impute_knn(n_neighbors=2)
    assert result.isna().sum().sum() == 0
