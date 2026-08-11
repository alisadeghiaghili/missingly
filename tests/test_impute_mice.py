"""Tests for impute_mice — single chain and multi-imputation behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.mi_contracts import ImputationResult
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

    def test_visit_sequence_controls_manual_fcs_sweep_order(self):
        """A requested FCS visit sequence determines the history insertion order."""
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0], "c": [1.0, 2.0, 3.0, 4.0]}
        )

        _, history = impute_mice(
            frame,
            max_iter=2,
            random_state=0,
            return_history=True,
            visit_sequence=["b", "a"],
        )

        assert list(history) == ["b", "a"]

    def test_visit_sequence_rejects_unknown_or_incomplete_targets(self, small_mixed_df):
        """An FCS visit sequence must contain every missing target exactly once."""
        with pytest.raises(ValueError, match="unknown"):
            impute_mice(small_mixed_df, visit_sequence=["age", "unknown"])
        with pytest.raises(ValueError, match="exactly once"):
            impute_mice(small_mixed_df, visit_sequence=["age", "score"])


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

    def test_history_chains_are_stochastic_and_reproducible(self, small_mixed_df):
        """History mode must retain posterior draws, not collapse to one trace."""
        first_results, first_histories = impute_mice(
            small_mixed_df,
            n_imputations=3,
            max_iter=5,
            random_state=17,
            return_history=True,
        )
        second_results, second_histories = impute_mice(
            small_mixed_df,
            n_imputations=3,
            max_iter=5,
            random_state=17,
            return_history=True,
        )

        assert any(
            first_histories[0][column] != first_histories[1][column]
            for column in first_histories[0]
        )
        assert first_histories == second_histories
        for first, second in zip(first_results, second_results):
            pd.testing.assert_frame_equal(first, second)

    def test_opt_in_result_contract_preserves_complete_mi_provenance(self, small_mixed_df):
        """Opt-in MICE output carries schema, seeds, chains, and tuple histories."""
        result = impute_mice(
            small_mixed_df,
            n_imputations=2,
            max_iter=3,
            random_state=17,
            return_result=True,
        )

        assert isinstance(result, ImputationResult)
        assert result.data.plan.seed_sequence == (17, 18)
        assert result.data.n_imputations == 2
        assert set(result.data.plan.methods) == {"age", "score", "category"}
        assert len(result.histories) == 2
        assert all(isinstance(trace, tuple) for history in result.histories for trace in history.values())
        assert result.validation["structural_validation_passed"] is True
        assert result.validation["convergence_assessed"] is False
        assert any("mice_convergence" in warning for warning in result.warnings)
        for imputed in result.data.imputations:
            assert not imputed.isna().any(axis=None)
            pd.testing.assert_series_equal(imputed.loc[0], small_mixed_df.loc[0])

    def test_result_contract_and_legacy_history_modes_are_exclusive(self, small_mixed_df):
        """The typed result already contains histories, so ambiguous output is rejected."""
        with pytest.raises(ValueError, match="return_result"):
            impute_mice(small_mixed_df, return_history=True, return_result=True)


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
