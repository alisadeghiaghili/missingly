"""Tests for missingly data manipulation functions.

Covers:
- replace_with_na / replace_with_na_all
- clean_names
- remove_empty  (including the thresh > 0 boundary fix)
- coalesce_columns
- miss_as_feature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.manipulation import (
    replace_with_na,
    replace_with_na_all,
    clean_names,
    remove_empty,
    coalesce_columns,
    miss_as_feature,
)


# ---------------------------------------------------------------------------
# replace_with_na
# ---------------------------------------------------------------------------

class TestReplaceWithNa:
    def test_scalar(self):
        df = pd.DataFrame({"A": [1, -99, 3]})
        result = replace_with_na(df, {"A": -99})
        assert pd.isna(result.loc[1, "A"])
        assert result.loc[0, "A"] == 1

    def test_list(self):
        df = pd.DataFrame({"A": [1, -99, -1, 3]})
        result = replace_with_na(df, {"A": [-99, -1]})
        assert pd.isna(result.loc[1, "A"])
        assert pd.isna(result.loc[2, "A"])

    def test_callable(self):
        df = pd.DataFrame({"A": [1, -99, 3]})
        result = replace_with_na(df, {"A": lambda x: x < 0})
        assert pd.isna(result.loc[1, "A"])

    def test_does_not_mutate(self):
        df = pd.DataFrame({"A": [1, -99, 3]})
        original = df.copy()
        replace_with_na(df, {"A": -99})
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# replace_with_na_all
# ---------------------------------------------------------------------------

class TestReplaceWithNaAll:
    def test_basic(self):
        df = pd.DataFrame({"A": [1, -99], "B": ["ok", "N/A"]})
        result = replace_with_na_all(df, lambda x: x in (-99, "N/A"))
        assert pd.isna(result.loc[1, "A"])
        assert pd.isna(result.loc[1, "B"])

    def test_does_not_mutate(self):
        df = pd.DataFrame({"A": [1, -99]})
        original = df.copy()
        replace_with_na_all(df, lambda x: x == -99)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# clean_names
# ---------------------------------------------------------------------------

class TestCleanNames:
    def test_basic_lowercase(self):
        df = pd.DataFrame(columns=["First Name", "Last Name"])
        assert clean_names(df).columns.tolist() == ["first_name", "last_name"]

    def test_special_chars(self):
        df = pd.DataFrame(columns=["Age#", "Income ($)"])
        result = clean_names(df)
        assert "#" not in result.columns[0]

    def test_persian_preserved(self):
        df = pd.DataFrame(columns=["درآمد ماهانه", "سن"])
        result = clean_names(df)
        # Persian chars must be preserved
        assert "درآمد" in result.columns[0]

    def test_duplicate_resolution(self):
        df = pd.DataFrame(columns=["A B", "A_B"])
        result = clean_names(df)
        assert len(set(result.columns)) == 2  # no duplicates

    def test_numeric_start(self):
        df = pd.DataFrame(columns=["1col"])
        assert not clean_names(df).columns[0][0].isdigit()

    def test_invalid_case_raises(self):
        df = pd.DataFrame(columns=["A"])
        with pytest.raises(ValueError, match="case"):
            clean_names(df, case="title")

    def test_invalid_sep_raises(self):
        df = pd.DataFrame(columns=["A"])
        with pytest.raises(ValueError, match="sep"):
            clean_names(df, sep="a")

    def test_does_not_mutate(self):
        df = pd.DataFrame(columns=["A B", "C D"])
        original_cols = df.columns.tolist()
        clean_names(df)
        assert df.columns.tolist() == original_cols

    def test_idempotent(self):
        df = pd.DataFrame(columns=["first_name", "last_name"])
        assert clean_names(df).columns.tolist() == ["first_name", "last_name"]


# ---------------------------------------------------------------------------
# remove_empty
# ---------------------------------------------------------------------------

class TestRemoveEmpty:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [np.nan, np.nan, np.nan],  # fully empty column
            "C": [1.0, 2.0, 3.0],
        })

    def test_drops_fully_empty_col(self, df):
        result = remove_empty(df)
        assert "B" not in result.columns
        assert "A" in result.columns

    def test_drops_fully_empty_row(self):
        df2 = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, np.nan]})
        result = remove_empty(df2, axis="rows")
        assert len(result) == 1  # row 1 was fully empty

    def test_thresh_col_drops_mostly_empty(self):
        # Column A has 1/3 ≈ 0.33 missing → dropped when thresh=0.3
        df2 = pd.DataFrame({"A": [np.nan, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})
        result = remove_empty(df2, thresh_col=0.3)
        assert "A" not in result.columns
        assert "B" in result.columns

    def test_thresh_col_keeps_below_threshold(self):
        # Column A has 1/3 ≈ 0.33 missing → kept when thresh=0.5
        df2 = pd.DataFrame({"A": [np.nan, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})
        result = remove_empty(df2, thresh_col=0.5)
        assert "A" in result.columns

    def test_thresh_col_zero_raises(self, df):
        """thresh_col=0 must raise ValueError (not silently drop all columns)."""
        with pytest.raises(ValueError, match="thresh_col"):
            remove_empty(df, thresh_col=0.0)

    def test_thresh_row_zero_raises(self, df):
        """thresh_row=0 must raise ValueError (not silently drop all rows)."""
        with pytest.raises(ValueError, match="thresh_row"):
            remove_empty(df, thresh_row=0.0)

    def test_invalid_axis_raises(self, df):
        with pytest.raises(ValueError, match="axis"):
            remove_empty(df, axis="diagonal")

    def test_sentinel_support(self, df):
        df2 = pd.DataFrame({"A": [-99, -99, -99], "B": [1, 2, 3]})
        result = remove_empty(df2, missing_values=[-99])
        assert "A" not in result.columns

    def test_does_not_mutate(self, df):
        original = df.copy()
        remove_empty(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# coalesce_columns
# ---------------------------------------------------------------------------

class TestCoalesceColumns:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "a": [1.0, np.nan, np.nan],
            "b": [np.nan, 2.0, np.nan],
            "c": [np.nan, np.nan, 3.0],
        })

    def test_fills_from_first_donor(self, df):
        result = coalesce_columns(df, "a", "b", "c")
        assert result.loc[1, "a"] == 2.0
        assert result.loc[2, "a"] == 3.0

    def test_original_value_preserved(self, df):
        result = coalesce_columns(df, "a", "b", "c")
        assert result.loc[0, "a"] == 1.0

    def test_remove_donors(self, df):
        result = coalesce_columns(df, "a", "b", "c", remove_donors=True)
        assert "b" not in result.columns
        assert "c" not in result.columns

    def test_no_donors_raises(self, df):
        with pytest.raises(ValueError, match="donor"):
            coalesce_columns(df, "a")

    def test_missing_column_raises(self, df):
        with pytest.raises(KeyError):
            coalesce_columns(df, "a", "z")

    def test_does_not_mutate(self, df):
        """coalesce_columns must not mutate the original DataFrame."""
        original_a = df["a"].copy()
        coalesce_columns(df, "a", "b", "c")
        np.testing.assert_array_equal(
            df["a"].values,
            original_a.values,
            strict=False,
        )


# ---------------------------------------------------------------------------
# miss_as_feature
# ---------------------------------------------------------------------------

class TestMissAsFeature:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "A": [1.0, np.nan, 3.0],
            "B": [np.nan, 2.0, 3.0],
            "C": [1.0, 2.0, 3.0],  # no missing
        })

    def test_indicator_columns_added(self, df):
        result = miss_as_feature(df)
        assert "A_NA" in result.columns
        assert "B_NA" in result.columns

    def test_no_indicator_for_complete_column(self, df):
        result = miss_as_feature(df)
        assert "C_NA" not in result.columns

    def test_indicator_values_correct(self, df):
        result = miss_as_feature(df)
        assert result.loc[1, "A_NA"] == 1
        assert result.loc[0, "A_NA"] == 0

    def test_keep_original_false(self, df):
        result = miss_as_feature(df, keep_original=False)
        assert "A" not in result.columns
        assert "A_NA" in result.columns

    def test_explicit_columns(self, df):
        result = miss_as_feature(df, columns=["C"])
        assert "C_NA" in result.columns  # forced even though C has no missing

    def test_missing_column_raises(self, df):
        with pytest.raises(KeyError):
            miss_as_feature(df, columns=["Z"])

    def test_sentinel_support(self):
        df2 = pd.DataFrame({"A": [1.0, -99.0, 3.0]})
        result = miss_as_feature(df2, missing_values=[-99])
        assert result.loc[1, "A_NA"] == 1
        assert result.loc[0, "A_NA"] == 0

    def test_column_order(self, df):
        """Indicator columns must be placed right after their source column."""
        result = miss_as_feature(df)
        cols = result.columns.tolist()
        assert cols.index("A_NA") == cols.index("A") + 1
        assert cols.index("B_NA") == cols.index("B") + 1

    def test_does_not_mutate(self, df):
        original = df.copy()
        miss_as_feature(df)
        pd.testing.assert_frame_equal(df, original)
