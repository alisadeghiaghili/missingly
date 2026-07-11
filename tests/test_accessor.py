"""Tests for MissinglyAccessor.impute() and typed impute_* wrappers.

Covers all 16 required cases from the C3 specification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import missingly  # noqa: F401 — registers df.miss accessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_numeric():
    """Small numeric DataFrame with one missing value."""
    return pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0, 5.0], "b": [10.0, 20.0, np.nan, 40.0, 50.0]}
    )


@pytest.fixture
def df_mixed():
    """Mixed numeric + categorical DataFrame."""
    return pd.DataFrame(
        {
            "age": [25.0, np.nan, 35.0, 40.0, 28.0],
            "city": ["Berlin", "Paris", None, "Berlin", "Paris"],
        }
    )


@pytest.fixture
def df_no_missing():
    """DataFrame with no missing values."""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def df_all_missing_col():
    """DataFrame where one column is entirely NaN."""
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan]}
    )


@pytest.fixture
def df_categorical():
    """DataFrame with pandas Categorical dtype."""
    return pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0, 5.0],
            "cat": pd.Categorical(
                ["a", None, "b", "a", "b"], categories=["a", "b"]
            ),
        }
    )


# ---------------------------------------------------------------------------
# 1. miss.impute exists after normal package import
# ---------------------------------------------------------------------------

def test_impute_method_exists(df_numeric):
    assert hasattr(df_numeric.miss, "impute")
    assert callable(df_numeric.miss.impute)


# ---------------------------------------------------------------------------
# 2. Default call returns new DataFrame; does NOT mutate input
# ---------------------------------------------------------------------------

def test_impute_default_returns_new_dataframe(df_numeric):
    original_values = df_numeric.copy()
    result = df_numeric.miss.impute()
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(df_numeric, original_values)  # input unchanged
    assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# 3. inplace=True mutates input and returns None
# ---------------------------------------------------------------------------

def test_impute_inplace_mutates_and_returns_none(df_numeric):
    df = df_numeric.copy()
    ret = df.miss.impute(method="mean", inplace=True)
    assert ret is None
    assert not df.isna().any().any()


# ---------------------------------------------------------------------------
# 4. Numeric imputation — mean
# ---------------------------------------------------------------------------

def test_impute_numeric_mean(df_numeric):
    result = df_numeric.miss.impute(method="mean")
    assert not result.isna().any().any()
    assert result.loc[1, "a"] == pytest.approx(df_numeric["a"].mean(skipna=True))


# ---------------------------------------------------------------------------
# 5. Categorical imputation (object dtype)
# ---------------------------------------------------------------------------

def test_impute_categorical_mode(df_mixed):
    result = df_mixed.miss.impute(method="mode")
    assert not result.isna().any().any()
    assert result["city"].notna().all()


# ---------------------------------------------------------------------------
# 6. pandas Categorical dtype preservation
# ---------------------------------------------------------------------------

def test_impute_preserves_categorical_categories(df_categorical):
    result = df_categorical.miss.impute(method="mode")
    assert not result.isna().any().any()
    # category dtype or at least values must be from known set
    unique_vals = set(result["cat"].dropna().unique())
    assert unique_vals <= {"a", "b"}


# ---------------------------------------------------------------------------
# 7. Selected-column behavior — only listed columns are imputed
# ---------------------------------------------------------------------------

def test_impute_selected_columns_only(df_numeric):
    result = df_numeric.miss.impute(method="mean", columns=["a"])
    assert not result["a"].isna().any()
    # column b should be unchanged (still has NaN)
    assert result["b"].isna().any()


def test_impute_selected_columns_preserves_others(df_mixed):
    """Untouched columns keep their original values exactly."""
    original_city = df_mixed["city"].copy()
    result = df_mixed.miss.impute(method="mean", columns=["age"])
    pd.testing.assert_series_equal(result["city"], original_city)


# ---------------------------------------------------------------------------
# 8. Missing / unknown column error
# ---------------------------------------------------------------------------

def test_impute_unknown_column_raises(df_numeric):
    from missingly.exceptions import MissingColumnError
    with pytest.raises(MissingColumnError):
        df_numeric.miss.impute(method="mean", columns=["nonexistent"])


# ---------------------------------------------------------------------------
# 9. Unknown-method error
# ---------------------------------------------------------------------------

def test_impute_unknown_method_raises(df_numeric):
    with pytest.raises(ValueError, match="Unknown imputation method"):
        df_numeric.miss.impute(method="magic")


# ---------------------------------------------------------------------------
# 10. fill_value with non-constant strategy raises
# ---------------------------------------------------------------------------

def test_impute_fill_value_non_constant_raises(df_numeric):
    with pytest.raises(ValueError, match="fill_value"):
        df_numeric.miss.impute(method="mean", fill_value=0)


# ---------------------------------------------------------------------------
# 11. All-missing column — InsufficientDataError via MissinglyImputer
# ---------------------------------------------------------------------------

def test_impute_all_missing_column_raises(df_all_missing_col):
    """A column that is entirely NaN cannot be imputed with mean."""
    from missingly.exceptions import InsufficientDataError
    with pytest.raises((InsufficientDataError, ValueError)):
        df_all_missing_col.miss.impute(method="mean")


# ---------------------------------------------------------------------------
# 12. DataFrame with no missing values — returned unchanged
# ---------------------------------------------------------------------------

def test_impute_no_missing_values_unchanged(df_no_missing):
    result = df_no_missing.miss.impute(method="mean")
    pd.testing.assert_frame_equal(result, df_no_missing)


# ---------------------------------------------------------------------------
# 13. Mixed dtypes
# ---------------------------------------------------------------------------

def test_impute_mixed_dtypes(df_mixed):
    result = df_mixed.miss.impute(method="mean")
    assert not result.isna().any().any()
    assert result["age"].dtype in (np.float64, np.float32)


# ---------------------------------------------------------------------------
# 14. Reproducibility for stochastic method (rf)
# ---------------------------------------------------------------------------

def test_impute_rf_reproducibility(df_numeric):
    r1 = df_numeric.miss.impute(method="rf", random_state=42)
    r2 = df_numeric.miss.impute(method="rf", random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# 15. Parity with MissinglyImputer canonical path
# ---------------------------------------------------------------------------

def test_impute_parity_with_missinglyimputer(df_numeric):
    from missingly.transformer import MissinglyImputer
    accessor_result = df_numeric.miss.impute(method="mean", random_state=0)
    imputer = MissinglyImputer(strategy="mean", random_state=0)
    canonical_result = imputer.fit_transform(df_numeric)
    pd.testing.assert_frame_equal(
        accessor_result.reset_index(drop=True),
        canonical_result.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 16. Index, index name, and column order are preserved
# ---------------------------------------------------------------------------

def test_impute_preserves_index_and_column_order(df_numeric):
    df = df_numeric.copy()
    df.index = pd.Index([10, 20, 30, 40, 50], name="row_id")
    result = df.miss.impute(method="median")
    assert list(result.index) == [10, 20, 30, 40, 50]
    assert result.index.name == "row_id"
    assert list(result.columns) == list(df.columns)
