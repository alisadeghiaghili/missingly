"""Tests for the missingly pandas accessor (df.miss.*).

Covers:
- Accessor registration and basic functionality
- Manipulation methods (replace_with_na, etc.)
- Imputation methods (impute_mean, impute_median, etc.)
- C3 unified impute() dispatcher (16 spec tests)
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

import missingly  # noqa: F401 — registers df.miss accessor


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
# C3 spec tests: df.miss.impute() (16 tests)
# ---------------------------------------------------------------------------


def test_impute_method_exists(df_numeric):
    """1. miss.impute exists after normal package import."""
    assert hasattr(df_numeric.miss, "impute")
    assert callable(df_numeric.miss.impute)


def test_impute_default_returns_new_dataframe(df_numeric):
    """2. Default call returns new DataFrame; does NOT mutate input."""
    original_values = df_numeric.copy()
    result = df_numeric.miss.impute()
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(df_numeric, original_values)
    assert not result.isna().any().any()


def test_impute_inplace_mutates_and_returns_none(df_numeric):
    """3. inplace=True mutates input and returns None."""
    df = df_numeric.copy()
    ret = df.miss.impute(method="mean", inplace=True)
    assert ret is None
    assert not df.isna().any().any()


def test_impute_numeric_mean(df_numeric):
    """4. Numeric imputation — mean."""
    result = df_numeric.miss.impute(method="mean")
    assert not result.isna().any().any()
    expected_age = df_numeric["age"].mean(skipna=True)
    assert result.loc[1, "age"] == pytest.approx(expected_age)


def test_impute_categorical_mode(df_mixed):
    """5. Categorical imputation (object dtype)."""
    result = df_mixed.miss.impute(method="mode")
    assert not result.isna().any().any()
    assert result["city"].notna().all()


def test_impute_preserves_categorical_categories(df_categorical):
    """6. pandas Categorical dtype — imputed value is from known categories."""
    result = df_categorical.miss.impute(method="mode")
    assert not result.isna().any().any()
    unique_vals = set(result["cat"].dropna().unique())
    assert unique_vals <= {"a", "b"}


def test_impute_selected_columns_only(df_numeric):
    """7. Selected-column behaviour — only listed columns are imputed."""
    result = df_numeric.miss.impute(method="mean", columns=["age"])
    assert not result["age"].isna().any()
    # income / score still have their original NaN values
    assert result["income"].isna().any() or result["score"].isna().any()


def test_impute_selected_columns_preserves_others(df_mixed):
    """8. Untouched columns keep their original values exactly."""
    original_city = df_mixed["city"].copy()
    result = df_mixed.miss.impute(method="mean", columns=["age"])
    pd.testing.assert_series_equal(result["city"], original_city)


def test_impute_unknown_column_raises(df_numeric):
    """9. Unknown column → MissingColumnError."""
    from missingly.exceptions import MissingColumnError
    with pytest.raises(MissingColumnError):
        df_numeric.miss.impute(method="mean", columns=["nonexistent"])


def test_impute_unknown_method_raises(df_numeric):
    """10. Unknown method → ValueError."""
    with pytest.raises(ValueError, match="Unknown imputation method"):
        df_numeric.miss.impute(method="magic")


def test_impute_fill_value_non_constant_raises(df_numeric):
    """11. fill_value with non-constant strategy → ValueError."""
    with pytest.raises(ValueError, match="fill_value"):
        df_numeric.miss.impute(method="mean", fill_value=0)


def test_impute_all_missing_column_raises(df_all_missing_col):
    """12. All-missing column → InsufficientDataError / ValueError."""
    from missingly.exceptions import InsufficientDataError
    with pytest.raises((InsufficientDataError, ValueError)):
        df_all_missing_col.miss.impute(method="mean")


def test_impute_no_missing_values_unchanged(df_no_missing):
    """13. DataFrame with no missing values → returned unchanged."""
    result = df_no_missing.miss.impute(method="mean")
    pd.testing.assert_frame_equal(result, df_no_missing)


def test_impute_mixed_dtypes(df_mixed):
    """14. Mixed dtypes."""
    result = df_mixed.miss.impute(method="mean")
    assert not result.isna().any().any()
    assert result["age"].dtype in (np.float64, np.float32)


def test_impute_rf_reproducibility(df_numeric):
    """15. Reproducibility for stochastic method (rf)."""
    r1 = df_numeric.miss.impute(method="rf", random_state=42)
    r2 = df_numeric.miss.impute(method="rf", random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_impute_parity_with_missinglyimputer(df_numeric):
    """16. Parity with canonical MissinglyImputer path."""
    from missingly.transformer import MissinglyImputer
    accessor_result = df_numeric.miss.impute(method="mean", random_state=0)
    imputer = MissinglyImputer(strategy="mean", random_state=0)
    canonical_result = imputer.fit_transform(df_numeric)
    pd.testing.assert_frame_equal(
        accessor_result.reset_index(drop=True),
        canonical_result.reset_index(drop=True),
    )


def test_impute_preserves_index_and_column_order(df_numeric):
    """Bonus: Index, index name, and column order are preserved."""
    df = df_numeric.copy()
    df.index = pd.Index([10, 20, 30, 40, 50], name="row_id")
    result = df.miss.impute(method="median")
    assert list(result.index) == [10, 20, 30, 40, 50]
    assert result.index.name == "row_id"
    assert list(result.columns) == list(df.columns)


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
        assert df_numeric.miss.n_miss() == 3

    def test_n_complete(self, df_numeric):
        assert df_numeric.miss.n_complete() == 12  # 15 cells - 3 missing

    def test_pct_miss(self, df_numeric):
        pct = df_numeric.miss.pct_miss()
        assert 0.0 <= pct <= 100.0

    def test_pct_complete(self, df_numeric):
        assert (df_numeric.miss.pct_miss() + df_numeric.miss.pct_complete()
                == pytest.approx(100.0))

    def test_miss_var_summary(self, df_numeric):
        result = df_numeric.miss.miss_var_summary()
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns
        assert "n_miss" in result.columns
        assert "pct_miss" in result.columns

    def test_miss_case_summary(self, df_numeric):
        result = df_numeric.miss.miss_case_summary()
        assert isinstance(result, pd.DataFrame)
        assert "case" in result.columns
        assert "n_miss" in result.columns

    def test_bind_shadow(self, df_numeric):
        result = df_numeric.miss.bind_shadow()
        assert result.shape[1] == df_numeric.shape[1] * 2

    def test_n_miss_no_missing(self, df_no_missing):
        assert df_no_missing.miss.n_miss() == 0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_mcar_test_returns_dict(self, df_numeric):
        result = df_numeric.miss.mcar_test()
        assert isinstance(result, dict)
        assert "chi_square" in result
        assert "df" in result
        assert "p_value" in result

    def test_mcar_test_raises_on_no_missing(self, df_no_missing):
        with pytest.raises(ValueError, match="missing"):
            df_no_missing.miss.mcar_test()

    def test_mar_mnar_test(self, df_numeric):
        result = df_numeric.miss.mar_mnar_test(target="age")
        assert isinstance(result, list)
        if result:
            assert len(result[0]) == 3


# ---------------------------------------------------------------------------
# Manipulation methods
# ---------------------------------------------------------------------------


class TestManipulation:
    def test_replace_with_na(self, df_no_missing):
        result = df_no_missing.miss.replace_with_na({"x": 2.0})
        assert result["x"].isna().sum() == 1

    def test_replace_with_na_does_not_mutate(self, df_no_missing):
        original = df_no_missing.copy(deep=True)
        df_no_missing.miss.replace_with_na({"x": 2.0})
        pd.testing.assert_frame_equal(df_no_missing, original)


# ---------------------------------------------------------------------------
# Imputation wrappers (impute_* methods)
# ---------------------------------------------------------------------------


class TestImputation:
    def test_impute_mean(self, df_numeric):
        result = df_numeric.miss.impute_mean()
        assert result.isna().sum().sum() == 0

    def test_impute_median(self, df_numeric):
        result = df_numeric.miss.impute_median()
        assert result.isna().sum().sum() == 0

    def test_impute_mode(self, df_mixed):
        result = df_mixed.miss.impute_mode()
        assert result.isna().sum().sum() == 0

    def test_impute_knn(self, df_numeric):
        result = df_numeric.miss.impute_knn(n_neighbors=2)
        assert result.isna().sum().sum() == 0

    def test_impute_mice(self, df_numeric):
        result = df_numeric.miss.impute_mice(max_iter=3)
        assert result.isna().sum().sum() == 0

    def test_impute_rf(self, df_numeric):
        result = df_numeric.miss.impute_rf()
        assert result.isna().sum().sum() == 0

    def test_impute_gb(self, df_numeric):
        result = df_numeric.miss.impute_gb()
        assert result.isna().sum().sum() == 0

    def test_imputation_does_not_mutate(self, df_numeric):
        original = df_numeric.copy(deep=True)
        df_numeric.miss.impute_mean()
        pd.testing.assert_frame_equal(df_numeric, original)


# ---------------------------------------------------------------------------
# Visualisation (smoke tests only)
# ---------------------------------------------------------------------------


class TestVisualisation:
    def test_matrix(self, df_numeric):
        ax = df_numeric.miss.matrix()
        assert hasattr(ax, "figure")

    def test_bar(self, df_numeric):
        ax = df_numeric.miss.bar()
        assert hasattr(ax, "figure")

    def test_heatmap(self, df_numeric):
        ax = df_numeric.miss.heatmap()
        assert hasattr(ax, "figure")

    def test_miss_var_pct(self, df_numeric):
        ax = df_numeric.miss.miss_var_pct()
        assert hasattr(ax, "figure")

    def test_vis_miss(self, df_numeric):
        ax = df_numeric.miss.vis_miss()
        assert hasattr(ax, "figure")


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


class TestChaining:
    def test_chaining_manipulation_and_summary(self, df_numeric):
        result = (
            df_numeric
            .miss.replace_with_na({"score": 85.0})
            .miss.n_miss()
        )
        assert isinstance(result, int)
        assert result > 0
