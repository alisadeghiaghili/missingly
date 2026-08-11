"""Unit tests for missingly.impute — error handling and input validation."""

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    FittedImputer,
    impute_gb,
    impute_knn,
    impute_mean,
    impute_median,
    impute_mice,
    impute_mode,
    impute_rf,
    make_imputer,
)
from missingly.exceptions import (
    InsufficientDataError,
    InvalidStrategyError,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, np.nan, 40.0, 50.0],
    })


@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 40.0, 50.0],
        "city": ["Berlin", "Munich", np.nan, "Berlin", "Hamburg"],
    })


@pytest.fixture
def df_all_missing_col():
    """One column is 100% NaN."""
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def df_single_row():
    return pd.DataFrame({"a": [np.nan], "b": [1.0]})


@pytest.fixture
def df_no_missing():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# Input validation — every public function must reject non-DataFrames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_reject_non_dataframe(func):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        func([1, 2, None])


@pytest.mark.parametrize("func", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_reject_empty_dataframe(func):
    with pytest.raises(ValueError, match="empty"):
        func(pd.DataFrame())


@pytest.mark.parametrize("func", [impute_mean, impute_median, impute_mode])
def test_simple_imputers_reject_series(func):
    with pytest.raises(TypeError):
        func(pd.Series([1.0, np.nan]))


# ---------------------------------------------------------------------------
# impute_mean
# ---------------------------------------------------------------------------

class TestImputeMean:
    def test_fills_numeric_with_mean(self, df_numeric):
        result = impute_mean(df_numeric)
        assert result["a"].isna().sum() == 0
        assert result.loc[1, "a"] == pytest.approx(3.25)  # mean of [1,3,4,5]

    def test_no_missing_returns_unchanged(self, df_no_missing):
        result = impute_mean(df_no_missing)
        pd.testing.assert_frame_equal(result, df_no_missing)

    def test_numeric_only_leaves_cat_unchanged(self, df_mixed):
        result = impute_mean(df_mixed, numeric_only=True)
        assert result["age"].isna().sum() == 0
        assert result["city"].isna().sum() == df_mixed["city"].isna().sum()

    def test_returns_copy_not_inplace(self, df_numeric):
        result = impute_mean(df_numeric)
        assert result is not df_numeric
        assert df_numeric["a"].isna().sum() > 0  # original unchanged


# ---------------------------------------------------------------------------
# impute_median
# ---------------------------------------------------------------------------

class TestImputeMedian:
    def test_fills_numeric_with_median(self, df_numeric):
        result = impute_median(df_numeric)
        assert result["a"].isna().sum() == 0

    def test_no_missing_returns_unchanged(self, df_no_missing):
        result = impute_median(df_no_missing)
        pd.testing.assert_frame_equal(result, df_no_missing)


# ---------------------------------------------------------------------------
# impute_mode
# ---------------------------------------------------------------------------

class TestImputeMode:
    def test_fills_categorical_with_mode(self, df_mixed):
        result = impute_mode(df_mixed)
        assert result["city"].isna().sum() == 0
        assert result.loc[2, "city"] == "Berlin"  # most frequent

    def test_fills_numeric_with_mode(self, df_numeric):
        result = impute_mode(df_numeric)
        assert result["a"].isna().sum() == 0


# ---------------------------------------------------------------------------
# impute_knn — validation
# ---------------------------------------------------------------------------

class TestImputeKnn:
    def test_invalid_metric_raises(self, df_numeric):
        with pytest.raises(InvalidStrategyError, match="metric"):
            impute_knn(df_numeric, metric="manhattan")

    def test_non_integer_n_neighbors_raises(self, df_numeric):
        with pytest.raises(TypeError, match="integer"):
            impute_knn(df_numeric, n_neighbors=2.5)  # type: ignore

    def test_zero_n_neighbors_raises(self, df_numeric):
        with pytest.raises(ValueError, match="positive"):
            impute_knn(df_numeric, n_neighbors=0)

    def test_euclidean_fills_all_nan(self, df_numeric):
        result = impute_knn(df_numeric, n_neighbors=2, metric="euclidean")
        assert result.isna().sum().sum() == 0

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            impute_knn([[1, 2], [None, 3]])


# ---------------------------------------------------------------------------
# impute_mice — validation
# ---------------------------------------------------------------------------

class TestImputeMice:
    def test_zero_n_imputations_raises(self, df_numeric):
        with pytest.raises(ValueError, match="positive"):
            impute_mice(df_numeric, n_imputations=0)

    def test_zero_max_iter_raises(self, df_numeric):
        with pytest.raises(ValueError, match="positive"):
            impute_mice(df_numeric, max_iter=0)

    def test_single_imputation_returns_dataframe(self, df_numeric):
        result = impute_mice(df_numeric, max_iter=2, random_state=0)
        assert isinstance(result, pd.DataFrame)
        assert result.isna().sum().sum() == 0

    def test_multiple_imputations_returns_list(self, df_numeric):
        results = impute_mice(df_numeric, n_imputations=3, max_iter=2, random_state=0)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, pd.DataFrame) for r in results)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            impute_mice("not a df")


# ---------------------------------------------------------------------------
# impute_rf — strict_mode
# ---------------------------------------------------------------------------

class TestImputeRf:
    def test_fills_numeric(self, df_numeric):
        result = impute_rf(df_numeric, random_state=0)
        assert result.isna().sum().sum() == 0

    def test_zero_max_iter_raises(self, df_numeric):
        with pytest.raises(ValueError, match="positive"):
            impute_rf(df_numeric, max_iter=0)

    def test_all_nan_col_raises_insufficient_data(self, df_all_missing_col):
        with pytest.raises(InsufficientDataError, match="b"):
            impute_rf(df_all_missing_col, random_state=0)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            impute_rf(None)


# ---------------------------------------------------------------------------
# impute_gb — strict_mode
# ---------------------------------------------------------------------------

class TestImputeGb:
    def test_fills_numeric(self, df_numeric):
        result = impute_gb(df_numeric, random_state=0)
        assert result.isna().sum().sum() == 0

    def test_all_nan_col_raises_insufficient_data(self, df_all_missing_col):
        with pytest.raises(InsufficientDataError, match="b"):
            impute_gb(df_all_missing_col, random_state=0)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            impute_gb({"a": [1, None]})


# ---------------------------------------------------------------------------
# FittedImputer
# ---------------------------------------------------------------------------

class TestFittedImputer:
    def test_invalid_strategy_raises_at_construction(self):
        with pytest.raises(InvalidStrategyError, match="strategy"):
            FittedImputer(strategy="bad")

    def test_is_fitted_false_before_fit(self):
        imp = make_imputer("mean")
        assert imp._is_fitted is False

    def test_is_fitted_true_after_fit(self, df_numeric):
        imp = make_imputer("mean")
        imp.fit(df_numeric)
        assert imp._is_fitted is True

    def test_transform_before_fit_raises_runtime_error(self, df_numeric):
        imp = make_imputer("mean")
        with pytest.raises(RuntimeError):
            imp.transform(df_numeric)

    def test_transform_non_dataframe_raises(self, df_numeric):
        imp = make_imputer("mean")
        imp.fit(df_numeric)
        with pytest.raises(TypeError):
            imp.transform([1, 2, 3])

    def test_fit_non_dataframe_raises(self):
        imp = make_imputer("mean")
        with pytest.raises(TypeError):
            imp.fit("not a df")

    def test_fit_transform_no_leakage(self, df_numeric):
        """Transform must use only training statistics."""
        train = df_numeric.copy()
        test = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        imp = make_imputer("mean")
        imp.fit(train)
        result = imp.transform(test)
        # Fill values must equal training means, not test means
        train_mean_a = train["a"].mean()
        assert result.loc[0, "a"] == pytest.approx(train_mean_a)

    def test_repr_unfitted(self):
        imp = make_imputer("median")
        assert "median" in repr(imp)

    def test_repr_fitted(self, df_numeric):
        imp = make_imputer("median")
        imp.fit(df_numeric)
        assert "median" in repr(imp)

    def test_make_imputer_returns_fitted_imputer(self):
        imp = make_imputer("mean")
        assert isinstance(imp, FittedImputer)
        assert imp.strategy == "mean"
