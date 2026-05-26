"""Tests for missingly.manipulation utilities.

Covers add_any_miss_var and bind_shadow_matrix (new functions),
plus smoke tests for the existing replace_with_na / replace_with_na_all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.manipulation import (
    add_any_miss_var,
    bind_shadow_matrix,
    replace_with_na,
    replace_with_na_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.nan, 2.0, 3.0],
        "c": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def complete_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def sentinel_df():
    return pd.DataFrame({"a": [1, -99, 3], "b": [4, 5, -99]})


# ---------------------------------------------------------------------------
# add_any_miss_var
# ---------------------------------------------------------------------------

class TestAddAnyMissVar:

    def test_returns_dataframe(self, simple_df):
        result = add_any_miss_var(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_one_column(self, simple_df):
        result = add_any_miss_var(simple_df)
        assert result.shape[1] == simple_df.shape[1] + 1

    def test_default_col_name(self, simple_df):
        result = add_any_miss_var(simple_df)
        assert "any_miss" in result.columns

    def test_custom_col_name(self, simple_df):
        result = add_any_miss_var(simple_df, col_name="has_na")
        assert "has_na" in result.columns

    def test_correct_boolean_values(self, simple_df):
        result = add_any_miss_var(simple_df)
        # row 0: b is NaN -> True
        # row 1: a is NaN -> True
        # row 2: no NaN  -> False
        expected = pd.Series([True, True, False], name="any_miss")
        pd.testing.assert_series_equal(result["any_miss"], expected)

    def test_all_complete_gives_all_false(self, complete_df):
        result = add_any_miss_var(complete_df)
        assert result["any_miss"].sum() == 0

    def test_all_missing_gives_all_true(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = add_any_miss_var(df)
        assert result["any_miss"].all()

    def test_sentinel_values(self, sentinel_df):
        result = add_any_miss_var(sentinel_df, missing_values=[-99])
        # row 0: no sentinel -> False
        # row 1: a=-99 -> True
        # row 2: b=-99 -> True
        expected = pd.Series([False, True, True], name="any_miss")
        pd.testing.assert_series_equal(result["any_miss"], expected)

    def test_does_not_modify_original(self, simple_df):
        original_cols = list(simple_df.columns)
        add_any_miss_var(simple_df)
        assert list(simple_df.columns) == original_cols

    def test_preserves_index(self, simple_df):
        df = simple_df.set_index(pd.Index([10, 20, 30]))
        result = add_any_miss_var(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_raises_if_col_name_exists(self, simple_df):
        df = simple_df.copy()
        df["any_miss"] = False
        with pytest.raises(ValueError, match="already exists"):
            add_any_miss_var(df)

    def test_single_column_df(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = add_any_miss_var(df)
        expected = pd.Series([False, True, False], name="any_miss")
        pd.testing.assert_series_equal(result["any_miss"], expected)

    def test_empty_df(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = add_any_miss_var(df)
        assert len(result) == 0
        assert "any_miss" in result.columns


# ---------------------------------------------------------------------------
# bind_shadow_matrix
# ---------------------------------------------------------------------------

class TestBindShadowMatrix:

    def test_returns_dataframe(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        assert result.shape == simple_df.shape

    def test_column_names(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        expected_cols = [f"{c}_NA" for c in simple_df.columns]
        assert list(result.columns) == expected_cols

    def test_boolean_dtype(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        assert all(result[c].dtype == bool for c in result.columns)

    def test_correct_true_positions(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        # a_NA: row 1 is True
        assert result["a_NA"].tolist() == [False, True, False]
        # b_NA: row 0 is True
        assert result["b_NA"].tolist() == [True, False, False]
        # c_NA: no missing
        assert result["c_NA"].tolist() == [False, False, False]

    def test_no_original_columns(self, simple_df):
        result = bind_shadow_matrix(simple_df)
        for col in simple_df.columns:
            assert col not in result.columns

    def test_complete_df_all_false(self, complete_df):
        result = bind_shadow_matrix(complete_df)
        assert result.values.sum() == 0

    def test_all_missing_all_true(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = bind_shadow_matrix(df)
        assert result.all().all()

    def test_sentinel_values(self, sentinel_df):
        result = bind_shadow_matrix(sentinel_df, missing_values=[-99])
        assert result["a_NA"].tolist() == [False, True, False]
        assert result["b_NA"].tolist() == [False, False, True]

    def test_preserves_index(self, simple_df):
        df = simple_df.set_index(pd.Index(["r1", "r2", "r3"]))
        result = bind_shadow_matrix(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_does_not_modify_original(self, simple_df):
        original = simple_df.copy()
        bind_shadow_matrix(simple_df)
        pd.testing.assert_frame_equal(simple_df, original)

    def test_empty_df(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = bind_shadow_matrix(df)
        assert result.shape == (0, 1)
        assert list(result.columns) == ["a_NA"]


# ---------------------------------------------------------------------------
# replace_with_na (smoke)
# ---------------------------------------------------------------------------

def test_replace_with_na_scalar():
    df = pd.DataFrame({"a": [1, -99, 3]})
    result = replace_with_na(df, replace={"a": -99})
    assert pd.isna(result.loc[1, "a"])
    assert result.loc[0, "a"] == 1


def test_replace_with_na_callable():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = replace_with_na(df, replace={"a": lambda x: x > 2})
    assert pd.isna(result.loc[2, "a"])
    assert not pd.isna(result.loc[0, "a"])


# ---------------------------------------------------------------------------
# replace_with_na_all (smoke)
# ---------------------------------------------------------------------------

def test_replace_with_na_all():
    df = pd.DataFrame({"a": [1, -99, 3], "b": [-99, 2, -99]})
    result = replace_with_na_all(df, condition=lambda x: x == -99)
    assert pd.isna(result.loc[1, "a"])
    assert pd.isna(result.loc[0, "b"])
    assert pd.isna(result.loc[2, "b"])
    assert result.loc[0, "a"] == 1


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_top_level():
    import missingly
    assert callable(missingly.add_any_miss_var)
    assert callable(missingly.bind_shadow_matrix)
    assert callable(missingly.replace_with_na)
    assert callable(missingly.replace_with_na_all)
