"""Tests for hot-deck imputation functions.

Property invariants tested per variant
---------------------------------------
* No missing values remain after imputation.
* All imputed values are drawn from the original observed pool
  (distribution-preservation property).
* CategoricalDtype is preserved.
* Index and columns are unchanged.
* Edge cases: single observed value, all-missing column (must raise),
  no missing data (no-op), mixed dtypes.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.hotdeck import (
    impute_hotdeck_random,
    impute_hotdeck_sequential,
    impute_hotdeck_weighted,
)
from missingly.transformer import MissinglyImputer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "age":  [25.0, np.nan, 35.0, 40.0, np.nan],
        "city": pd.Categorical(["Paris", "Lyon", np.nan, "Paris", "Lyon"]),
    })


@pytest.fixture
def no_missing_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})


# ---------------------------------------------------------------------------
# impute_hotdeck_random
# ---------------------------------------------------------------------------

class TestHotdeckRandom:
    def test_no_missing_after_imputation(self, mixed_df):
        result = impute_hotdeck_random(mixed_df, random_state=0)
        assert result.isnull().sum().sum() == 0

    def test_imputed_values_in_donor_pool(self, numeric_df):
        donors = {1.0, 3.0, 5.0}
        result = impute_hotdeck_random(numeric_df, random_state=42)
        imputed = result.loc[numeric_df["a"].isna(), "a"]
        assert set(imputed).issubset(donors)

    def test_categorical_dtype_preserved(self, mixed_df):
        result = impute_hotdeck_random(mixed_df, random_state=0)
        assert isinstance(result["city"].dtype, pd.CategoricalDtype)

    def test_columns_unchanged(self, mixed_df):
        result = impute_hotdeck_random(mixed_df, random_state=0)
        assert list(result.columns) == list(mixed_df.columns)

    def test_index_preserved(self, mixed_df):
        result = impute_hotdeck_random(mixed_df, random_state=0)
        pd.testing.assert_index_equal(result.index, mixed_df.index)

    def test_no_missing_input_is_noop(self, no_missing_df):
        result = impute_hotdeck_random(no_missing_df, random_state=0)
        pd.testing.assert_frame_equal(result, no_missing_df)

    def test_single_observed_value(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, 7.0]})
        result = impute_hotdeck_random(df, random_state=0)
        assert (result["a"] == 7.0).all()

    def test_all_missing_raises(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="no observed values"):
            impute_hotdeck_random(df)

    def test_reproducible_with_seed(self, mixed_df):
        r1 = impute_hotdeck_random(mixed_df, random_state=99)
        r2 = impute_hotdeck_random(mixed_df, random_state=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_returns_dataframe(self, mixed_df):
        assert isinstance(impute_hotdeck_random(mixed_df), pd.DataFrame)


# ---------------------------------------------------------------------------
# impute_hotdeck_sequential
# ---------------------------------------------------------------------------

class TestHotdeckSequential:
    def test_forward_locf(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
        result = impute_hotdeck_sequential(df, direction="forward")
        assert result["a"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_backward_nocb(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, 3.0, 4.0]})
        result = impute_hotdeck_sequential(df, direction="backward")
        assert result["a"].tolist() == [3.0, 3.0, 3.0, 4.0]

    def test_nearest_fills_all(self, numeric_df):
        result = impute_hotdeck_sequential(numeric_df, direction="nearest")
        assert result.isnull().sum().sum() == 0

    def test_forward_leaves_leading_nan(self):
        df = pd.DataFrame({"a": [np.nan, 2.0, np.nan]})
        result = impute_hotdeck_sequential(df, direction="forward")
        assert pd.isna(result["a"].iloc[0])
        assert result["a"].iloc[2] == 2.0

    def test_categorical_dtype_preserved(self, mixed_df):
        result = impute_hotdeck_sequential(mixed_df, direction="nearest")
        assert isinstance(result["city"].dtype, pd.CategoricalDtype)

    def test_invalid_direction_raises(self, numeric_df):
        with pytest.raises(ValueError, match="direction"):
            impute_hotdeck_sequential(numeric_df, direction="sideways")

    def test_no_missing_input_is_noop(self, no_missing_df):
        result = impute_hotdeck_sequential(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)

    def test_returns_dataframe(self, numeric_df):
        assert isinstance(impute_hotdeck_sequential(numeric_df), pd.DataFrame)


# ---------------------------------------------------------------------------
# impute_hotdeck_weighted
# ---------------------------------------------------------------------------

class TestHotdeckWeighted:
    def test_no_missing_after_imputation(self, mixed_df):
        result = impute_hotdeck_weighted(mixed_df, random_state=0)
        assert result.isnull().sum().sum() == 0

    def test_imputed_values_in_donor_pool(self, numeric_df):
        donors = {1.0, 3.0, 5.0}
        result = impute_hotdeck_weighted(numeric_df, n_donors=3, random_state=0)
        imputed = result.loc[numeric_df["a"].isna(), "a"]
        assert set(imputed).issubset(donors)

    def test_categorical_dtype_preserved(self, mixed_df):
        result = impute_hotdeck_weighted(mixed_df, random_state=0)
        assert isinstance(result["city"].dtype, pd.CategoricalDtype)

    def test_n_donors_1_is_deterministic(self, numeric_df):
        r1 = impute_hotdeck_weighted(numeric_df, n_donors=1, random_state=0)
        r2 = impute_hotdeck_weighted(numeric_df, n_donors=1, random_state=99)
        # With n_donors=1 the nearest neighbour is always chosen regardless of seed
        pd.testing.assert_frame_equal(r1, r2)

    def test_all_missing_raises(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        with pytest.raises(ValueError, match="no observed values"):
            impute_hotdeck_weighted(df)

    def test_returns_dataframe(self, mixed_df):
        assert isinstance(impute_hotdeck_weighted(mixed_df), pd.DataFrame)

    def test_index_preserved(self, mixed_df):
        result = impute_hotdeck_weighted(mixed_df, random_state=0)
        pd.testing.assert_index_equal(result.index, mixed_df.index)


# ---------------------------------------------------------------------------
# MissinglyImputer(strategy='hotdeck') — all three methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["random", "sequential", "weighted"])
class TestMissinglyImputerHotdeck:
    def test_no_missing_after_fit_transform(self, method, mixed_df):
        imp = MissinglyImputer(strategy="hotdeck", hotdeck_method=method)
        result = imp.fit_transform(mixed_df)
        assert result.isnull().sum().sum() == 0

    def test_categorical_dtype_preserved(self, method, mixed_df):
        imp = MissinglyImputer(strategy="hotdeck", hotdeck_method=method)
        result = imp.fit_transform(mixed_df)
        assert isinstance(result["city"].dtype, pd.CategoricalDtype)

    def test_returns_dataframe(self, method, mixed_df):
        imp = MissinglyImputer(strategy="hotdeck", hotdeck_method=method)
        assert isinstance(imp.fit_transform(mixed_df), pd.DataFrame)

    def test_transform_uses_train_donors_only(self, method):
        """Donors for transform() must come from training data, not test data."""
        train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        test = pd.DataFrame({"a": [np.nan]})
        imp = MissinglyImputer(strategy="hotdeck", hotdeck_method=method)
        imp.fit(train)
        result = imp.transform(test)
        assert result["a"].iloc[0] in {10.0, 20.0, 30.0}
