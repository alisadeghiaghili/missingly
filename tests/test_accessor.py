"""Tests for the missingly pandas accessor (df.miss.*).

Covers:
- Accessor registration and basic functionality
- Manipulation methods (replace_with_na, miss_as_feature, etc.)
- Imputation methods (impute_mean, impute_median, etc.)
- Summary methods (n_miss, pct_miss, miss_var_summary, etc.)
- Statistical tests (mcar_test, mar_mnar_test)
- Visualisation methods (smoke tests only)
- Chaining capability
- Mutation safety
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import missingly  # noqa: F401 — registers the .miss accessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_numeric():
    """Small numeric DataFrame with missing values."""
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 40.0, 30.0],
        "income": [50_000.0, 60_000.0, np.nan, 80_000.0, 70_000.0],
        "score": [85.0, 90.0, 78.0, np.nan, 88.0],
    })


@pytest.fixture
def df_mixed():
    """DataFrame with numeric and categorical columns."""
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 40.0, 30.0],
        "city": ["Berlin", "Munich", np.nan, "Berlin", "Hamburg"],
        "score": [85.0, 90.0, 78.0, np.nan, 88.0],
    })


@pytest.fixture
def df_no_missing():
    """Complete DataFrame with no missing values."""
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": ["x", "y", "z", "x", "y"],
    })


# ---------------------------------------------------------------------------
# Accessor registration
# ---------------------------------------------------------------------------


class TestAccessorRegistration:
    def test_accessor_is_registered(self):
        """The .miss accessor should be available on DataFrames."""
        df = pd.DataFrame({"a": [1.0]})
        assert hasattr(df, "miss")

    def test_accessor_returns_MissinglyAccessor(self):
        """df.miss should return a MissinglyAccessor instance."""
        from missingly.accessor import MissinglyAccessor
        df = pd.DataFrame({"a": [1.0]})
        assert isinstance(df.miss, MissinglyAccessor)


# ---------------------------------------------------------------------------
# Summary methods
# ---------------------------------------------------------------------------


class TestSummary:
    def test_n_miss(self, df_numeric):
        """n_miss should return the total number of missing values."""
        assert df_numeric.miss.n_miss() == 3

    def test_n_complete(self, df_numeric):
        """n_complete should return the total number of non-missing values."""
        assert df_numeric.miss.n_complete() == 12  # 15 cells - 3 missing

    def test_pct_miss(self, df_numeric):
        """pct_miss should return a percentage between 0 and 100."""
        pct = df_numeric.miss.pct_miss()
        assert 0.0 <= pct <= 100.0

    def test_pct_complete(self, df_numeric):
        """pct_complete should complement pct_miss."""
        assert (df_numeric.miss.pct_miss() + df_numeric.miss.pct_complete()
                == pytest.approx(100.0))

    def test_miss_var_summary(self, df_numeric):
        """miss_var_summary should return a DataFrame with expected columns."""
        result = df_numeric.miss.miss_var_summary()
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns
        assert "n_miss" in result.columns
        assert "pct_miss" in result.columns

    def test_miss_case_summary(self, df_numeric):
        """miss_case_summary should return a DataFrame with expected columns."""
        result = df_numeric.miss.miss_case_summary()
        assert isinstance(result, pd.DataFrame)
        assert "case" in result.columns
        assert "n_miss" in result.columns

    def test_bind_shadow(self, df_numeric):
        """bind_shadow should double the number of columns."""
        result = df_numeric.miss.bind_shadow()
        assert result.shape[1] == df_numeric.shape[1] * 2

    def test_n_miss_no_missing(self, df_no_missing):
        """n_miss should return 0 for a complete DataFrame."""
        assert df_no_missing.miss.n_miss() == 0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_mcar_test_returns_dict(self, df_numeric):
        """mcar_test should return a dict with expected keys."""
        result = df_numeric.miss.mcar_test()
        assert isinstance(result, dict)
        assert "chi_square" in result
        assert "df" in result
        assert "p_value" in result

    def test_mcar_test_raises_on_no_missing(self, df_no_missing):
        """mcar_test should raise ValueError on all-complete DataFrame."""
        with pytest.raises(ValueError, match="missing"):
            df_no_missing.miss.mcar_test()

    def test_mar_mnar_test(self, df_numeric):
        """mar_mnar_test should return a list of tuples."""
        result = df_numeric.miss.mar_mnar_test(target="age")
        assert isinstance(result, list)
        if result:  # may be empty if no missing patterns
            assert len(result[0]) == 3  # (feature, LRT, p_value)


# ---------------------------------------------------------------------------
# Manipulation methods
# ---------------------------------------------------------------------------


class TestManipulation:
    def test_replace_with_na(self, df_no_missing):
        """replace_with_na should replace specified values with NaN."""
        result = df_no_missing.miss.replace_with_na({"a": 2.0})
        assert result["a"].isna().sum() == 1
        assert result.loc[1, "a"] is np.nan or pd.isna(result.loc[1, "a"])

    def test_replace_with_na_does_not_mutate(self, df_no_missing):
        """replace_with_na should return a new DataFrame."""
        original = df_no_missing.copy(deep=True)
        df_no_missing.miss.replace_with_na({"a": 2.0})
        pd.testing.assert_frame_equal(df_no_missing, original)

    def test_add_any_miss_var(self, df_numeric):
        """add_any_miss_var should add a boolean column."""
        result = df_numeric.miss.add_any_miss_var()
        assert "any_miss" in result.columns
        assert result["any_miss"].dtype == bool

    def test_bind_shadow_matrix(self, df_numeric):
        """bind_shadow_matrix should return only the shadow matrix."""
        result = df_numeric.miss.bind_shadow_matrix()
        assert result.shape[1] == df_numeric.shape[1]
        assert all(col.endswith("_NA") for col in result.columns)


# ---------------------------------------------------------------------------
# Imputation methods
# ---------------------------------------------------------------------------


class TestImputation:
    def test_impute_mean(self, df_numeric):
        """impute_mean should fill missing values."""
        result = df_numeric.miss.impute_mean()
        assert result.isna().sum().sum() == 0

    def test_impute_median(self, df_numeric):
        """impute_median should fill missing values."""
        result = df_numeric.miss.impute_median()
        assert result.isna().sum().sum() == 0

    def test_impute_mode(self, df_mixed):
        """impute_mode should fill missing values."""
        result = df_mixed.miss.impute_mode()
        assert result.isna().sum().sum() == 0

    def test_impute_knn(self, df_numeric):
        """impute_knn should fill missing values."""
        result = df_numeric.miss.impute_knn(n_neighbors=2)
        assert result.isna().sum().sum() == 0

    def test_impute_mice(self, df_numeric):
        """impute_mice should fill missing values."""
        result = df_numeric.miss.impute_mice(max_iter=3)
        assert result.isna().sum().sum() == 0

    def test_impute_rf(self, df_numeric):
        """impute_rf should fill missing values."""
        result = df_numeric.miss.impute_rf()
        assert result.isna().sum().sum() == 0

    def test_impute_gb(self, df_numeric):
        """impute_gb should fill missing values."""
        result = df_numeric.miss.impute_gb()
        assert result.isna().sum().sum() == 0

    def test_imputation_does_not_mutate(self, df_numeric):
        """Imputation methods should not mutate the original DataFrame."""
        original = df_numeric.copy(deep=True)
        df_numeric.miss.impute_mean()
        pd.testing.assert_frame_equal(df_numeric, original)


# ---------------------------------------------------------------------------
# Visualisation (smoke tests only)
# ---------------------------------------------------------------------------


class TestVisualisation:
    def test_matrix(self, df_numeric):
        """matrix should return an Axes object."""
        ax = df_numeric.miss.matrix()
        assert hasattr(ax, "figure")

    def test_bar(self, df_numeric):
        """bar should return an Axes object."""
        ax = df_numeric.miss.bar()
        assert hasattr(ax, "figure")

    def test_heatmap(self, df_numeric):
        """heatmap should return an Axes object."""
        ax = df_numeric.miss.heatmap()
        assert hasattr(ax, "figure")

    def test_miss_var_pct(self, df_numeric):
        """miss_var_pct should return an Axes object."""
        ax = df_numeric.miss.miss_var_pct()
        assert hasattr(ax, "figure")

    def test_vis_miss(self, df_numeric):
        """vis_miss should return an Axes object."""
        ax = df_numeric.miss.vis_miss()
        assert hasattr(ax, "figure")


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


class TestChaining:
    def test_chaining_manipulation_and_summary(self, df_numeric):
        """Methods should be chainable."""
        result = (
            df_numeric
            .miss.add_any_miss_var()
            .miss.n_miss()
        )
        # n_miss counts including the new indicator column
        assert isinstance(result, int)
