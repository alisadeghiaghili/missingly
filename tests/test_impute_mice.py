"""Tests for impute_mice — single chain and multi-imputation behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

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

    def test_predictor_matrix_limits_manual_fcs_features(self):
        """A target row in predictor_matrix selects only its enabled columns."""
        class RecordingRegressor(BaseEstimator, RegressorMixin):
            """Record the number of selected FCS predictors in each fit."""

            fitted_widths: list[int] = []

            def fit(self, X, y):
                """Record the selected predictor width.

                Parameters
                ----------
                X : np.ndarray
                    Conditional-model features.
                y : np.ndarray
                    Observed target values.

                Returns
                -------
                RecordingRegressor
                    This fitted deterministic estimator.
                """
                type(self).fitted_widths.append(X.shape[1])
                return self

            def predict(self, X):
                """Return deterministic conditional predictions.

                Parameters
                ----------
                X : np.ndarray
                    Conditional-model features.

                Returns
                -------
                np.ndarray
                    One zero prediction per row.
                """
                return np.zeros(X.shape[0])

        frame = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0], "z": [5.0, 6.0, 7.0, 8.0], "y": [1.0, np.nan, 3.0, 4.0]}
        )
        matrix = pd.DataFrame(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
            index=["x", "z", "y"],
            columns=["x", "z", "y"],
            dtype=bool,
        )

        result = impute_mice(frame, max_iter=2, estimator=RecordingRegressor(), predictor_matrix=matrix)

        assert not result["y"].isna().any()
        assert RecordingRegressor.fitted_widths == [1, 1]

    def test_predictor_matrix_rejects_misaligned_or_self_predictors(self, small_mixed_df):
        """Predictor matrices require exact labels and a zero diagonal."""
        misaligned = pd.DataFrame([[False]], index=["age"], columns=["age"])
        with pytest.raises(ValueError, match="labels"):
            impute_mice(small_mixed_df, predictor_matrix=misaligned)

        labels = list(small_mixed_df.columns)
        self_predictor = pd.DataFrame(True, index=labels, columns=labels)
        with pytest.raises(ValueError, match="diagonal"):
            impute_mice(small_mixed_df, predictor_matrix=self_predictor)

    def test_where_mask_imputes_only_explicitly_enabled_missing_cells(self):
        """where can retain structural missingness without altering observed cells."""
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]}
        )
        where = frame.isna()
        where.loc[1, "a"] = False

        result = impute_mice(frame, max_iter=2, random_state=0, where=where)

        assert pd.isna(result.loc[1, "a"])
        assert not pd.isna(result.loc[2, "b"])
        assert result.loc[0, "a"] == frame.loc[0, "a"]

    def test_where_mask_rejects_misaligned_or_observed_cells(self, small_mixed_df):
        """where must align exactly and cannot request over-imputation."""
        bad_shape = pd.DataFrame(False, index=small_mixed_df.index, columns=["age"])
        with pytest.raises(ValueError, match="align"):
            impute_mice(small_mixed_df, where=bad_shape)

        overimpute = small_mixed_df.isna()
        overimpute.loc[0, "age"] = True
        with pytest.raises(ValueError, match="observed"):
            impute_mice(small_mixed_df, where=overimpute)

    def test_where_mask_rejects_nullable_boolean_missing_values(self):
        """Three-valued where masks fail before they can weaken eligibility."""
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
        where = frame.isna().astype("boolean")
        where.loc[1, "a"] = pd.NA

        with pytest.raises(ValueError, match="where must not contain missing values"):
            impute_mice(frame, max_iter=2, where=where)

    def test_typed_result_preserves_partial_structural_missingness(self):
        """Typed MICE results retain where-disabled cells across every chain."""
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]}
        )
        where = frame.isna().astype("boolean")
        where.loc[1, "a"] = False
        original = frame.copy(deep=True)
        original_where = where.copy(deep=True)

        result = impute_mice(
            frame,
            max_iter=2,
            n_imputations=2,
            random_state=0,
            where=where,
            return_result=True,
        )

        assert isinstance(result, ImputationResult)
        assert result.data.n_imputations == 2
        pd.testing.assert_frame_equal(result.data.schema.original_mask, frame.isna())
        pd.testing.assert_frame_equal(
            result.data.schema.imputation_mask,
            frame.isna() & where,
        )
        assert result.validation["structural_missingness_retained"] is True
        assert result.data.plan.provenance["where_mask"] == "original_missing_subset"
        for completed in result.data.imputations:
            assert pd.isna(completed.loc[1, "a"])
            assert not pd.isna(completed.loc[2, "b"])
            assert completed.loc[0, "a"] == frame.loc[0, "a"]
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(where, original_where)

    def test_typed_result_all_false_where_retains_every_original_missing_cell(self):
        """An all-false where mask yields valid typed chains with structural NAs."""
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
        where = pd.DataFrame(False, index=frame.index, columns=frame.columns)

        result = impute_mice(
            frame,
            max_iter=2,
            n_imputations=2,
            random_state=0,
            where=where,
            return_result=True,
        )

        assert not result.data.schema.imputation_mask.any(axis=None)
        for completed in result.data.imputations:
            pd.testing.assert_frame_equal(completed, frame)

    def test_typed_result_discloses_default_conditional_target_contract(self):
        """Typed MICE metadata distinguishes numeric and categorical approximations."""
        frame = pd.DataFrame(
            {
                "income": [10.0, np.nan, 30.0, 40.0],
                "tier": pd.Categorical(["low", "high", None, "low"], ordered=True),
            }
        )

        result = impute_mice(
            frame,
            max_iter=2,
            n_imputations=2,
            random_state=0,
            return_result=True,
        )

        assert result.data.plan.methods == {
            "income": "bayesian_ridge_numeric",
            "tier": "ordinal_encoded_bayesian_ridge_approximation",
        }
        assert result.data.plan.provenance["conditional_model_contract"] == (
            "numeric=bayesian_ridge_posterior; "
            "categorical=ordinal_encoded_bayesian_ridge_approximation"
        )
        assert any("not a validated multinomial or ordinal FCS kernel" in warning
                   for warning in result.warnings)

    def test_typed_result_discloses_custom_categorical_approximation(self):
        """Custom estimators still disclose ordinal encoding for categorical targets."""
        class ZeroRegressor(BaseEstimator, RegressorMixin):
            """Return deterministic zero predictions for a contract test."""

            def fit(self, X, y):
                """Fit the deterministic test estimator.

                Parameters
                ----------
                X : np.ndarray
                    Conditional-model feature matrix.
                y : np.ndarray
                    Observed target values.

                Returns
                -------
                ZeroRegressor
                    This fitted estimator.
                """
                return self

            def predict(self, X):
                """Return a zero-valued prediction for each row.

                Parameters
                ----------
                X : np.ndarray
                    Conditional-model feature matrix.

                Returns
                -------
                np.ndarray
                    One deterministic prediction per input row.
                """
                return np.zeros(X.shape[0])

        frame = pd.DataFrame(
            {
                "income": [10.0, np.nan, 30.0, 40.0],
                "tier": pd.Categorical(["low", "high", None, "low"], ordered=True),
            }
        )
        original = frame.copy(deep=True)

        result = impute_mice(
            frame,
            estimator=ZeroRegressor(),
            max_iter=2,
            n_imputations=2,
            random_state=0,
            return_result=True,
        )

        assert result.data.plan.methods == {
            "income": "custom_estimator_numeric",
            "tier": "ordinal_encoded_custom_estimator_approximation",
        }
        assert result.data.plan.provenance["conditional_model_contract"] == (
            "numeric=caller_supplied_custom_estimator; "
            "categorical=ordinal_encoded_custom_estimator_approximation"
        )
        assert any("not a validated multinomial or ordinal FCS kernel" in warning
                   for warning in result.warnings)
        for completed in result.data.imputations:
            assert completed["tier"].dtype == frame["tier"].dtype
            assert completed.loc[2, "tier"] in frame["tier"].cat.categories
        pd.testing.assert_frame_equal(frame, original)

    def test_typed_result_marks_absent_categorical_contract_not_applicable(self):
        """Numeric-only MICE provenance does not claim a categorical approximation."""
        frame = pd.DataFrame(
            {
                "income": [10.0, np.nan, 30.0, 40.0],
                "age": [20.0, 30.0, 40.0, 50.0],
            }
        )

        result = impute_mice(
            frame,
            max_iter=2,
            n_imputations=2,
            random_state=0,
            return_result=True,
        )

        assert result.data.plan.provenance["conditional_model_contract"] == (
            "numeric=bayesian_ridge_posterior; categorical=not_applicable"
        )
        assert not any("multinomial or ordinal FCS kernel" in warning
                       for warning in result.warnings)

    def test_numeric_bounds_clip_only_imputed_cells(self):
        """Declared bounds constrain draws without changing observed data."""
        class HighRegressor(BaseEstimator, RegressorMixin):
            """Return an out-of-domain prediction to exercise bound clipping."""

            def fit(self, X, y):
                """Fit the deterministic test double.

                Parameters
                ----------
                X : np.ndarray
                    Predictor matrix.
                y : np.ndarray
                    Observed target values.

                Returns
                -------
                HighRegressor
                    This fitted test double.
                """
                return self

            def predict(self, X):
                """Return a deliberately out-of-range draw.

                Parameters
                ----------
                X : np.ndarray
                    Predictor matrix.

                Returns
                -------
                np.ndarray
                    One high value per prediction row.
                """
                return np.full(X.shape[0], 99.0)

        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "age": [20.0, np.nan, 40.0, 50.0]})
        result = impute_mice(frame, estimator=HighRegressor(), bounds={"age": (0.0, 80.0)})

        assert result.loc[1, "age"] == 80.0
        assert result.loc[0, "age"] == 20.0

    def test_numeric_bounds_reject_unknown_or_invalid_intervals(self, small_mixed_df):
        """Bounds must name numeric schema columns and have finite ascending limits."""
        with pytest.raises(ValueError, match="unknown"):
            impute_mice(small_mixed_df, bounds={"unknown": (0.0, 1.0)})
        with pytest.raises(ValueError, match="lower"):
            impute_mice(small_mixed_df, bounds={"age": (2.0, 1.0)})


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
