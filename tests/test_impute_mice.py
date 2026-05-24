"""Tests for impute_mice — single chain and multi-imputation behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_mice


@pytest.fixture()
def small_mixed_df() -> pd.DataFrame:
    """Tiny DataFrame with numeric + categorical columns and NaN values."""
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
    def test_returns_dataframe(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert isinstance(result, pd.DataFrame)

    def test_same_shape(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert result.shape == small_mixed_df.shape

    def test_same_columns(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert list(result.columns) == list(small_mixed_df.columns)

    def test_no_missing_values(self, small_mixed_df):
        result = impute_mice(small_mixed_df, max_iter=2, random_state=0)
        assert result.isnull().sum().sum() == 0

    def test_explicit_n_imputations_1(self, small_mixed_df):
        result = impute_mice(small_mixed_df, n_imputations=1, max_iter=2)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 2 — n_imputations=3: returns a list of DataFrames
# ---------------------------------------------------------------------------

class TestMultipleImputation:
    @pytest.fixture()
    def results(self, small_mixed_df):
        return impute_mice(small_mixed_df, n_imputations=3, max_iter=5, random_state=0)

    @pytest.fixture()
    def results_alt_seed(self, small_mixed_df):
        return impute_mice(small_mixed_df, n_imputations=3, max_iter=5, random_state=100)

    def test_returns_list(self, results):
        assert isinstance(results, list)

    def test_list_length(self, results):
        assert len(results) == 3

    def test_all_elements_are_dataframes(self, results):
        for i, df in enumerate(results):
            assert isinstance(df, pd.DataFrame), f"Element {i} is not a pd.DataFrame."

    def test_all_have_correct_columns(self, results, small_mixed_df):
        for i, df in enumerate(results):
            assert list(df.columns) == list(small_mixed_df.columns)

    def test_all_have_correct_shape(self, results, small_mixed_df):
        for i, df in enumerate(results):
            assert df.shape == small_mixed_df.shape

    def test_no_missing_values_in_any_dataset(self, results):
        for i, df in enumerate(results):
            assert df.isnull().sum().sum() == 0

    def test_chains_differ(self, results, results_alt_seed):
        """Two runs with different base seeds should produce distinct numeric values."""
        numeric_cols = results[0].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            assert not results[0][numeric_cols].equals(results_alt_seed[0][numeric_cols]), (
                "Runs with different random_state should produce distinct imputations."
            )

    def test_within_run_chains_differ(self, results):
        """Within the same run, chain 0 and chain 1 should differ (different seeds)."""
        numeric_cols = results[0].select_dtypes(include=[np.number]).columns.tolist()
        if len(results) >= 2 and numeric_cols:
            assert not results[0][numeric_cols].equals(results[1][numeric_cols]), (
                "Chain 0 (seed=0) and chain 1 (seed=1) should produce distinct imputations."
            )


# ---------------------------------------------------------------------------
# Test 3 — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_n_imputations_raises(self, small_mixed_df):
        with pytest.raises(ValueError):
            impute_mice(small_mixed_df, n_imputations=0)

    def test_no_missing_passthrough(self):
        df_clean = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = impute_mice(df_clean, max_iter=2)
        assert result.isnull().sum().sum() == 0
        assert result.shape == df_clean.shape
