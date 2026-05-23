"""Tests for impute_mice — single chain and multi-imputation behaviour.

Keep fixtures small so the test suite stays fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_mice


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_mixed_df() -> pd.DataFrame:
    """Tiny DataFrame with numeric + categorical columns and NaN values.

    Shape: 8 rows × 3 columns.
    * ``age``      — float with one NaN
    * ``score``    — float with one NaN
    * ``category`` — object (categorical) with one NaN
    """
    return pd.DataFrame(
        {
            "age":      [25.0, np.nan, 35.0, 40.0, 22.0, 31.0, 28.0, 45.0],
            "score":    [7.2,  8.1,   np.nan, 6.5, 9.0,  7.8,  8.3,  6.1],
            "category": ["A",  "B",   "A",    None, "B",  "A",  "B",  "A"],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — n_imputations=1 (default): backward-compatible single DataFrame
# ---------------------------------------------------------------------------

class TestSingleImputation:
    """impute_mice(df) with n_imputations=1 must behave exactly as before."""

    def test_returns_dataframe(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert isinstance(result, pd.DataFrame), (
            "n_imputations=1 must return a pd.DataFrame, not a list."
        )

    def test_same_shape(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert result.shape == small_mixed_df.shape

    def test_same_columns(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert list(result.columns) == list(small_mixed_df.columns)

    def test_no_missing_values(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert result.isnull().sum().sum() == 0, (
            "Result must contain zero NaN values."
        )

    def test_explicit_n_imputations_1(self, small_mixed_df):
        """Passing n_imputations=1 explicitly must also return a DataFrame."""
        result = impute_mice(small_mixed_df, n_imputations=1, max_iter=2)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 2 — n_imputations=3: returns a list of DataFrames
# ---------------------------------------------------------------------------

class TestMultipleImputation:
    """impute_mice(df, n_imputations=3) must return a list of 3 DataFrames."""

    @pytest.fixture()
    def results(self, small_mixed_df):
        return impute_mice(small_mixed_df, n_imputations=3, max_iter=2, random_state=0)

    def test_returns_list(self, results):
        assert isinstance(results, list), (
            "n_imputations > 1 must return a list."
        )

    def test_list_length(self, results):
        assert len(results) == 3, "List length must equal n_imputations."

    def test_all_elements_are_dataframes(self, results):
        for i, df in enumerate(results):
            assert isinstance(df, pd.DataFrame), (
                f"Element {i} is not a pd.DataFrame."
            )

    def test_all_have_correct_columns(self, results, small_mixed_df):
        for i, df in enumerate(results):
            assert list(df.columns) == list(small_mixed_df.columns), (
                f"Element {i} has wrong columns."
            )

    def test_all_have_correct_shape(self, results, small_mixed_df):
        for i, df in enumerate(results):
            assert df.shape == small_mixed_df.shape, (
                f"Element {i} has shape {df.shape}, expected {small_mixed_df.shape}."
            )

    def test_no_missing_values_in_any_dataset(self, results):
        for i, df in enumerate(results):
            n_missing = df.isnull().sum().sum()
            assert n_missing == 0, (
                f"Imputed dataset {i} still contains {n_missing} missing values."
            )

    def test_chains_differ(self, results):
        """Independent chains should produce at least slightly different numeric values."""
        numeric_cols = results[0].select_dtypes(include=[np.number]).columns.tolist()
        if len(results) >= 2 and numeric_cols:
            assert not results[0][numeric_cols].equals(results[1][numeric_cols]), (
                "Independent chains with different seeds should produce distinct imputations."
            )


# ---------------------------------------------------------------------------
# Test 3 — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_n_imputations_raises(self, small_mixed_df):
        with pytest.raises(ValueError, match="n_imputations must be >= 1"):
            impute_mice(small_mixed_df, n_imputations=0)

    def test_no_missing_data_passthrough(self):
        """DataFrame with no NaN should pass through without error."""
        df_clean = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = impute_mice(df_clean, max_iter=2)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0

    def test_n_imputations_2_returns_list_length_2(self, small_mixed_df):
        results = impute_mice(small_mixed_df, n_imputations=2, max_iter=2)
        assert isinstance(results, list)
        assert len(results) == 2
