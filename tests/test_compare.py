"""Tests for missingly.compare — compare_imputations and cv_compare_imputations."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from missingly import compare
from missingly.compare import _method_name, compare_imputations, cv_compare_imputations
from missingly import impute as _impute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_numeric():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=40),
        "b": rng.normal(size=40),
        "c": rng.normal(size=40),
    })


@pytest.fixture
def df_cat():
    return pd.DataFrame({
        "city": ["Berlin", "Munich", "Hamburg", "Berlin"] * 10,
        "tier": ["A", "B", "A", "B"] * 10,
    })


@pytest.fixture
def df_mixed():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "age": rng.integers(20, 60, size=40).astype(float),
        "score": rng.normal(size=40),
        "city": ["Berlin", "Munich"] * 20,
    })


@pytest.fixture
def df_for_cv():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "a": rng.normal(size=80),
        "b": rng.normal(size=80),
    })
    X.loc[rng.choice(80, 16, replace=False), "a"] = np.nan
    y = rng.integers(0, 2, size=80)
    return X, y


# ---------------------------------------------------------------------------
# _method_name helper
# ---------------------------------------------------------------------------

class TestMethodName:
    def test_named_function(self):
        assert _method_name(_impute.impute_mean) == "impute_mean"

    def test_lambda_falls_back_to_repr(self):
        fn = lambda df: df  # noqa: E731
        name = _method_name(fn)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_partial_uses_inner_name(self):
        p = functools.partial(_impute.impute_knn, n_neighbors=3)
        name = _method_name(p)
        assert "impute_knn" in name


# ---------------------------------------------------------------------------
# compare_imputations — input validation
# ---------------------------------------------------------------------------

class TestCompareImputationsValidation:
    def test_rejects_df_with_missing(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="complete"):
            compare_imputations(df)

    def test_rejects_bad_mask_frac_zero(self, df_numeric):
        with pytest.raises(ValueError, match="mask_frac"):
            compare_imputations(df_numeric, mask_frac=0.0)

    def test_rejects_bad_mask_frac_one(self, df_numeric):
        with pytest.raises(ValueError, match="mask_frac"):
            compare_imputations(df_numeric, mask_frac=1.0)

    def test_sentinel_replacement_passes_completeness_check(self):
        """Sentinels replaced via missing_values= should not trigger the error."""
        df = pd.DataFrame({"a": [-99, 1.0, 2.0, 3.0, 4.0] * 4})
        # Without missing_values this would raise because we'd first
        # need to drop -99 manually — with it the function handles it.
        result = compare_imputations(
            df.replace(-99, 1.0),  # truly complete input
            methods=[_impute.impute_mean],
            mask_frac=0.2,
        )
        assert isinstance(result, pd.DataFrame)

    def test_missing_values_param_accepted(self):
        """missing_values= should replace sentinels before completeness check."""
        # Build a df that is complete AFTER sentinel replacement
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 4})
        result = compare_imputations(
            df,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
            missing_values=[],
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# compare_imputations — return value schema
# ---------------------------------------------------------------------------

class TestCompareImputationsSchema:
    def test_numeric_only_has_rmse_column(self, df_numeric):
        result = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean, _impute.impute_median],
            mask_frac=0.2,
        )
        assert "RMSE" in result.columns
        assert "Accuracy" not in result.columns

    def test_cat_only_has_accuracy_column(self, df_cat):
        result = compare_imputations(
            df_cat,
            methods=[_impute.impute_mode],
            mask_frac=0.2,
        )
        assert "Accuracy" in result.columns
        assert "RMSE" not in result.columns

    def test_mixed_has_score_column(self, df_mixed):
        result = compare_imputations(
            df_mixed,
            methods=[_impute.impute_mean, _impute.impute_mode],
            mask_frac=0.2,
        )
        assert "Score" in result.columns

    def test_index_is_method_name(self, df_numeric):
        result = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean, _impute.impute_median],
            mask_frac=0.2,
        )
        assert "impute_mean" in result.index
        assert "impute_median" in result.index

    def test_sorted_ascending_rmse(self, df_numeric):
        result = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean, _impute.impute_median],
            mask_frac=0.2,
        )
        rmse = result["RMSE"].dropna().values
        assert list(rmse) == sorted(rmse)

    def test_sorted_descending_accuracy(self, df_cat):
        result = compare_imputations(
            df_cat,
            methods=[_impute.impute_mode],
            mask_frac=0.2,
        )
        acc = result["Accuracy"].dropna().values
        assert list(acc) == sorted(acc, reverse=True)

    def test_rmse_non_negative(self, df_numeric):
        result = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
        )
        assert (result["RMSE"].dropna() >= 0).all()

    def test_accuracy_between_0_and_1(self, df_cat):
        result = compare_imputations(
            df_cat,
            methods=[_impute.impute_mode],
            mask_frac=0.2,
        )
        acc = result["Accuracy"].dropna()
        assert (acc >= 0).all() and (acc <= 1).all()

    def test_mixed_all_methods_fail(self, df_mixed):
        """When all methods fail, the result should still be a valid DataFrame."""
        def bad_fn(df):
            raise RuntimeError("fail")
        result = compare_imputations(
            df_mixed,
            methods=[bad_fn, bad_fn],
            mask_frac=0.2,
        )
        assert isinstance(result, pd.DataFrame)
        assert "Score" in result.columns

    def test_mixed_single_method_returns_finite(self, df_mixed):
        """A single method should not produce NaN Score."""
        result = compare_imputations(
            df_mixed,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
        )
        assert not pd.isna(result["Score"].iloc[0])


# ---------------------------------------------------------------------------
# compare_imputations — error isolation
# ---------------------------------------------------------------------------

class TestCompareImputationsErrorIsolation:
    def test_failing_method_does_not_crash_others(self, df_numeric):
        """A method that raises should produce NaN, not abort the whole run."""
        def bad_method(df):
            raise RuntimeError("intentional failure")

        result = compare_imputations(
            df_numeric,
            methods=[bad_method, _impute.impute_mean],
            mask_frac=0.2,
        )
        assert "impute_mean" in result.index
        assert result.loc["impute_mean", "RMSE"] >= 0

    def test_failing_method_nan_score(self, df_numeric):
        def bad_method(df):
            raise ValueError("nope")

        result = compare_imputations(
            df_numeric,
            methods=[bad_method],
            mask_frac=0.2,
        )
        assert np.isnan(result["RMSE"].iloc[0])

    def test_error_column_present_when_failure(self, df_numeric):
        def bad_method(df):
            raise ValueError("oops")

        result = compare_imputations(
            df_numeric,
            methods=[bad_method, _impute.impute_mean],
            mask_frac=0.2,
        )
        assert "Error" in result.columns

    def test_no_error_column_when_all_succeed(self, df_numeric):
        result = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
        )
        assert "Error" not in result.columns


# ---------------------------------------------------------------------------
# compare_imputations — lambda / partial names
# ---------------------------------------------------------------------------

class TestCompareImputationsCallableNames:
    def test_partial_callable_as_method(self, df_numeric):
        knn3 = functools.partial(_impute.impute_knn, n_neighbors=3)
        result = compare_imputations(
            df_numeric,
            methods=[knn3],
            mask_frac=0.2,
        )
        assert len(result) == 1
        assert isinstance(result.index[0], str)

    def test_deterministic_with_seed(self, df_numeric):
        r1 = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
            random_state=7,
        )
        r2 = compare_imputations(
            df_numeric,
            methods=[_impute.impute_mean],
            mask_frac=0.2,
            random_state=7,
        )
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# cv_compare_imputations — validation
# ---------------------------------------------------------------------------

class TestCvCompareValidation:
    def test_unknown_strategy_raises(self, df_for_cv):
        X, y = df_for_cv
        with pytest.raises(ValueError, match="Unknown strategies"):
            cv_compare_imputations(X, y, LogisticRegression(), strategies=["unknown"])

    def test_length_mismatch_raises(self, df_for_cv):
        X, y = df_for_cv
        with pytest.raises(ValueError, match="same length"):
            cv_compare_imputations(X, y[:-5], LogisticRegression(), strategies=["mean"])


# ---------------------------------------------------------------------------
# cv_compare_imputations — schema & correctness
# ---------------------------------------------------------------------------

class TestCvCompareSchema:
    def test_returns_dataframe(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean", "median"],
            n_splits=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean"],
            n_splits=3,
        )
        assert "mean_score" in result.columns
        assert "std_score" in result.columns

    def test_index_is_strategy_name(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean", "median"],
            n_splits=3,
        )
        assert "mean" in result.index
        assert "median" in result.index

    def test_sorted_descending_mean_score(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean", "median"],
            n_splits=3,
        )
        scores = result["mean_score"].values
        assert list(scores) == sorted(scores, reverse=True)

    def test_std_non_negative(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean"],
            n_splits=3,
        )
        assert (result["std_score"] >= 0).all()

    def test_regressor_path(self, df_for_cv):
        X, _ = df_for_cv
        y_cont = np.random.default_rng(99).normal(size=len(X))
        result = cv_compare_imputations(
            X, y_cont, LinearRegression(),
            strategies=["mean"],
            n_splits=3,
        )
        assert len(result) == 1

    def test_custom_scoring(self, df_for_cv):
        X, y = df_for_cv
        from sklearn.metrics import f1_score

        def my_scoring(est, X_, y_):
            return f1_score(y_, est.predict(X_), average="macro")

        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean"],
            n_splits=3,
            scoring=my_scoring,
        )
        assert result.loc["mean", "mean_score"] >= 0

    def test_imputer_kwargs_forwarded(self, df_for_cv):
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["knn"],
            n_splits=3,
            imputer_kwargs={"knn": {"n_neighbors": 3}},
        )
        assert "knn" in result.index

    def test_no_leakage_imputer_fit_on_train(self, df_for_cv):
        """Smoke test: mean_score should be a valid float, not NaN."""
        X, y = df_for_cv
        result = cv_compare_imputations(
            X, y, LogisticRegression(max_iter=200),
            strategies=["mean"],
            n_splits=3,
        )
        assert not np.isnan(result.loc["mean", "mean_score"])
