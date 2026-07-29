"""Tests for diagnostics summary functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import (
    bind_shadow,
    n_miss,
    pct_miss,
    n_complete,
    pct_complete,
    bind_shadow,
)


@pytest.fixture()
def simple_df():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, 2.0, np.nan, 4.0, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture()
def no_missing_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture()
def all_missing_df():
    return pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})


class TestScalarHelpers:
    def test_n_miss(self, simple_df):
        assert n_miss(simple_df) == 4

    def test_pct_miss(self, simple_df):
        expected = 4 / simple_df.size * 100
        assert abs(pct_miss(simple_df) - expected) < 1e-9

    def test_n_complete(self, simple_df):
        assert n_complete(simple_df) == simple_df.size - 4

    def test_pct_complete_sums_to_100(self, simple_df):
        assert abs(pct_complete(simple_df) + pct_miss(simple_df) - 100) < 1e-9

    def test_n_miss_no_missing(self, no_missing_df):
        assert n_miss(no_missing_df) == 0

    def test_n_miss_all_missing(self, all_missing_df):
        assert n_miss(all_missing_df) == all_missing_df.size


class TestMissVarSummary:
    def test_returns_dataframe(self, simple_df):
        assert isinstance(miss_var_summary(simple_df), pd.DataFrame)

    def test_required_columns(self, simple_df):
        result = miss_var_summary(simple_df)
        assert {"variable", "n_miss", "pct_miss"}.issubset(result.columns)

    def test_counts_correct(self, simple_df):
        result = miss_var_summary(simple_df).set_index("variable")
        assert result.loc["a", "n_miss"] == 2
        assert result.loc["b", "n_miss"] == 2
        assert result.loc["c", "n_miss"] == 0

    def test_no_missing(self, no_missing_df):
        assert (miss_var_summary(no_missing_df)["n_miss"] == 0).all()


class TestMissCaseSummary:
    def test_returns_dataframe(self, simple_df):
        assert isinstance(miss_case_summary(simple_df), pd.DataFrame)

    def test_row_count(self, simple_df):
        assert len(miss_case_summary(simple_df)) == len(simple_df)

    def test_required_columns(self, simple_df):
        result = miss_case_summary(simple_df)
        assert {"case", "n_miss", "pct_miss"}.issubset(result.columns)


class TestBindShadow:
    def test_doubles_columns(self, simple_df):
        result = bind_shadow(simple_df)
        assert result.shape[1] == simple_df.shape[1] * 2

    def test_shadow_column_names(self, simple_df):
        result = bind_shadow(simple_df)
        for col in simple_df.columns:
            assert f"{col}_NA" in result.columns

    def test_shadow_values_bool(self, simple_df):
        result = bind_shadow(simple_df)
        shadow_cols = [c for c in result.columns if c.endswith("_NA")]
        for col in shadow_cols:
            assert result[col].isin([True, False]).all()
