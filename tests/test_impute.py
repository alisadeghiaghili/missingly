"""Tests for MissinglyImputer and imputation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import MissinglyImputer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_numeric_df():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, 2.0, np.nan, 4.0, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture()
def simple_categorical_df():
    return pd.DataFrame(
        {
            "cat": ["a", None, "b", None, "a"],
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )


@pytest.fixture()
def no_missing_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------

class TestMissinglyImputerInit:
    def test_default_strategy(self):
        imp = MissinglyImputer()
        assert imp is not None

    def test_mean_strategy(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean")
        result = imp.fit_transform(simple_numeric_df)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0

    def test_median_strategy(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="median")
        result = imp.fit_transform(simple_numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_mode_strategy(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mode")
        result = imp.fit_transform(simple_numeric_df)
        assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Fit / transform separation
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_returns_self(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean")
        returned = imp.fit(simple_numeric_df)
        assert returned is imp

    def test_transform_no_missing(self, simple_numeric_df, no_missing_df):
        imp = MissinglyImputer(strategy="mean")
        imp.fit(simple_numeric_df)
        result = imp.transform(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)

    def test_preserves_columns(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean")
        result = imp.fit_transform(simple_numeric_df)
        assert list(result.columns) == list(simple_numeric_df.columns)

    def test_preserves_index(self, simple_numeric_df):
        simple_numeric_df.index = list(range(10, 15))
        imp = MissinglyImputer(strategy="mean")
        result = imp.fit_transform(simple_numeric_df)
        assert list(result.index) == list(simple_numeric_df.index)


# ---------------------------------------------------------------------------
# Categorical support
# ---------------------------------------------------------------------------

class TestCategoricalImputation:
    def test_mode_fills_object_col(self, simple_categorical_df):
        imp = MissinglyImputer(strategy="mode")
        result = imp.fit_transform(simple_categorical_df)
        assert result["cat"].isnull().sum() == 0

    def test_pandas_categorical_dtype_preserved(self):
        df = pd.DataFrame({"c": pd.Categorical(["x", None, "y", None, "x"])})
        imp = MissinglyImputer(strategy="mode")
        result = imp.fit_transform(df)
        assert result["c"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

class TestColumnSelection:
    def test_selected_columns_only(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean", columns=["a"])
        result = imp.fit_transform(simple_numeric_df)
        # Column 'a' should have no missing
        assert result["a"].isnull().sum() == 0
        # Column 'b' should still have missing (not touched)
        assert result["b"].isnull().sum() == simple_numeric_df["b"].isnull().sum()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_missing_passthrough(self, no_missing_df):
        imp = MissinglyImputer(strategy="mean")
        result = imp.fit_transform(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            MissinglyImputer(strategy="invalid_xyz").fit_transform(
                pd.DataFrame({"a": [1.0, np.nan]})
            )
