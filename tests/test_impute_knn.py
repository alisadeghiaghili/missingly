"""Tests for impute_knn — correctness, safety warnings, and edge cases."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_knn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """Small all-numeric DataFrame with two missing values."""
    return pd.DataFrame(
        {
            "age":    [25.0, np.nan, 35.0, 40.0, 22.0, 31.0],
            "score":  [7.2,  8.1,   np.nan, 6.5,  9.0,  7.8],
            "income": [50.0, 60.0,  55.0,  np.nan, 45.0, 70.0],
        }
    )


@pytest.fixture()
def heavy_cat_df() -> pd.DataFrame:
    """DataFrame dominated by categoricals: 2 categorical + 1 numeric.

    Used to trigger the heavy-categorical safety warning when
    n_neighbors > 5.
    """
    return pd.DataFrame(
        {
            "colour":   ["red", "blue", "green", None,    "red",  "blue",
                         "green", "red",  "blue",  "green"],
            "category": ["A",   "B",    "A",     "B",     None,   "A",
                         "B",    "A",    "B",     "A"],
            "value":    [1.0,   2.0,    np.nan,  4.0,     5.0,    6.0,
                         7.0,   8.0,    9.0,     10.0],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — basic correctness on a numeric DataFrame
# ---------------------------------------------------------------------------

class TestNumericImputation:
    def test_returns_dataframe(self, numeric_df):
        result = impute_knn(numeric_df, n_neighbors=3)
        assert isinstance(result, pd.DataFrame)

    def test_same_shape(self, numeric_df):
        result = impute_knn(numeric_df, n_neighbors=3)
        assert result.shape == numeric_df.shape

    def test_same_columns(self, numeric_df):
        result = impute_knn(numeric_df, n_neighbors=3)
        assert list(result.columns) == list(numeric_df.columns)

    def test_no_missing_values(self, numeric_df):
        result = impute_knn(numeric_df, n_neighbors=3)
        assert result.isnull().sum().sum() == 0

    def test_no_warning_on_numeric_df(self, numeric_df):
        """Numeric-only DataFrame must not trigger the categorical warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            impute_knn(numeric_df, n_neighbors=10)


# ---------------------------------------------------------------------------
# Test 2 — heavy-categorical safety warning
# ---------------------------------------------------------------------------

class TestHeavyCategoricalWarning:
    def test_warns_when_heavy_cat_and_large_k(self, heavy_cat_df):
        """Must emit UserWarning matching 'KNN imputation may be unreliable'."""
        with pytest.warns(UserWarning, match="KNN imputation may be unreliable"):
            impute_knn(heavy_cat_df, n_neighbors=10)

    def test_no_warning_with_default_k(self, heavy_cat_df):
        """n_neighbors=5 (default) must NOT trigger the warning, even on heavy-cat data."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            impute_knn(heavy_cat_df, n_neighbors=5)
        cat_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "KNN imputation may be unreliable" in str(w.message)
        ]
        assert len(cat_warnings) == 0, (
            "n_neighbors=5 should not trigger the heavy-categorical warning."
        )

    def test_result_is_complete(self, heavy_cat_df):
        """Even when a warning fires the result must have zero NaN values."""
        with pytest.warns(UserWarning):
            result = impute_knn(heavy_cat_df, n_neighbors=10)
        assert result.isnull().sum().sum() == 0

    def test_result_has_correct_shape(self, heavy_cat_df):
        with pytest.warns(UserWarning):
            result = impute_knn(heavy_cat_df, n_neighbors=10)
        assert result.shape == heavy_cat_df.shape


# ---------------------------------------------------------------------------
# Test 3 — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_missing_data_passthrough(self):
        """DataFrame with no NaN should pass through without error or warning."""
        df_clean = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = impute_knn(df_clean)
        assert result.isnull().sum().sum() == 0

    def test_single_cat_single_num_no_warning(self):
        """Equal number of cat and numeric columns: no warning (cat not strictly > num)."""
        df = pd.DataFrame(
            {"label": ["A", "B", None, "A"], "value": [1.0, 2.0, 3.0, np.nan]}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            impute_knn(df, n_neighbors=10)
        cat_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "KNN imputation may be unreliable" in str(w.message)
        ]
        assert len(cat_warnings) == 0
