"""Tests for imputation functions and FittedImputer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    impute_mean,
    impute_median,
    impute_mode,
    impute_knn,
    FittedImputer,
    make_imputer,
)
from missingly.transformer import MissinglyImputer


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


class TestImputeMean:
    def test_fills_numeric(self, simple_numeric_df):
        result = impute_mean(simple_numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_preserves_columns(self, simple_numeric_df):
        result = impute_mean(simple_numeric_df)
        assert list(result.columns) == list(simple_numeric_df.columns)

    def test_preserves_index(self, simple_numeric_df):
        result = impute_mean(simple_numeric_df)
        assert list(result.index) == list(simple_numeric_df.index)

    def test_no_missing_passthrough(self, no_missing_df):
        result = impute_mean(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)


class TestImputeMedian:
    def test_fills_numeric(self, simple_numeric_df):
        result = impute_median(simple_numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_no_missing_passthrough(self, no_missing_df):
        result = impute_median(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)


class TestImputeMode:
    def test_fills_categorical(self, simple_categorical_df):
        result = impute_mode(simple_categorical_df)
        assert result.isnull().sum().sum() == 0

    def test_fills_numeric(self, simple_numeric_df):
        result = impute_mode(simple_numeric_df)
        assert result.isnull().sum().sum() == 0


class TestFittedImputer:
    def test_fit_returns_self(self, simple_numeric_df):
        imp = FittedImputer(strategy="mean")
        assert imp.fit(simple_numeric_df) is imp

    def test_fit_transform_no_missing(self, simple_numeric_df):
        result = FittedImputer(strategy="mean").fit_transform(simple_numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_invalid_strategy_raises(self):
        with pytest.raises(Exception):
            FittedImputer(strategy="invalid_xyz")

    def test_transform_before_fit_raises(self, no_missing_df):
        with pytest.raises(RuntimeError):
            FittedImputer(strategy="mean").transform(no_missing_df)

    def test_make_imputer_returns_fitted_imputer(self):
        assert isinstance(make_imputer("median"), FittedImputer)


class TestMissinglyImputer:
    def test_mean_strategy(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean")
        result = imp.fit_transform(simple_numeric_df)
        assert result.isnull().sum().sum() == 0

    def test_fit_returns_self(self, simple_numeric_df):
        imp = MissinglyImputer(strategy="mean")
        assert imp.fit(simple_numeric_df) is imp

    def test_preserves_columns(self, simple_numeric_df):
        result = MissinglyImputer(strategy="mean").fit_transform(simple_numeric_df)
        assert list(result.columns) == list(simple_numeric_df.columns)

    def test_preserves_index(self, simple_numeric_df):
        simple_numeric_df.index = list(range(10, 15))
        result = MissinglyImputer(strategy="mean").fit_transform(simple_numeric_df)
        assert list(result.index) == list(simple_numeric_df.index)

    def test_categorical_mode(self, simple_categorical_df):
        result = MissinglyImputer(strategy="mode").fit_transform(simple_categorical_df)
        assert result["cat"].isnull().sum() == 0

    def test_invalid_strategy_raises(self):
        with pytest.raises(Exception):
            MissinglyImputer(strategy="invalid_xyz").fit_transform(
                pd.DataFrame({"a": [1.0, np.nan]})
            )

    def test_no_missing_passthrough(self, no_missing_df):
        result = MissinglyImputer(strategy="mean").fit_transform(no_missing_df)
        pd.testing.assert_frame_equal(result, no_missing_df)
