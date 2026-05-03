"""Tests for missingly.manipulation module."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from missingly import manipulation
from missingly.manipulation import (
    clean_names,
    remove_empty,
    coalesce_columns,
    miss_as_feature,
)


@pytest.fixture
def sample_df():
    """Numeric and string dataframe with sentinel missing values."""
    return pd.DataFrame({
        'A': [1, -99, 3, 4],
        'B': ['x', 'y', 'N/A', 'z'],
        'C': [10.0, 20.0, 30.0, -99.0],
    })


@pytest.fixture
def nan_df():
    """Simple NaN dataframe for remove_empty / coalesce / miss_as_feature."""
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0],
        'B': [np.nan, np.nan, np.nan],
        'C': [1.0, np.nan, 3.0],
    })


# ---------------------------------------------------------------------------
# replace_with_na
# ---------------------------------------------------------------------------

def test_replace_with_na(sample_df):
    """Scalar replacement inserts NaN at the correct positions."""
    result = manipulation.replace_with_na(
        sample_df, replace={'A': -99, 'B': 'N/A', 'C': -99.0}
    )
    assert pd.isnull(result.loc[1, 'A'])
    assert pd.isnull(result.loc[2, 'B'])
    assert pd.isnull(result.loc[3, 'C'])
    assert not pd.isnull(result.loc[0, 'A'])
    assert not pd.isnull(result.loc[1, 'B'])


def test_replace_with_na_callable(sample_df):
    """Callable replacement inserts NaN where the predicate is True."""
    result = manipulation.replace_with_na(
        sample_df, replace={'A': lambda x: x < 0}
    )
    assert pd.isnull(result.loc[1, 'A'])
    assert not pd.isnull(result.loc[0, 'A'])


def test_replace_with_na_all(sample_df):
    """replace_with_na_all applies the condition across every cell."""
    result = manipulation.replace_with_na_all(
        sample_df, condition=lambda x: x == -99 or x == 'N/A'
    )
    assert pd.isnull(result.loc[1, 'A'])
    assert pd.isnull(result.loc[2, 'B'])
    assert pd.isnull(result.loc[3, 'C'])
    assert not pd.isnull(result.loc[0, 'A'])
    assert not pd.isnull(result.loc[1, 'B'])


# ---------------------------------------------------------------------------
# clean_names
# ---------------------------------------------------------------------------

def test_clean_names_spaces():
    """Spaces are replaced with underscores and result is lowercase."""
    df = pd.DataFrame(columns=['First Name', 'Last Name'])
    assert clean_names(df).columns.tolist() == ['first_name', 'last_name']


def test_clean_names_special_chars():
    """Special characters are replaced and consecutive seps are collapsed."""
    df = pd.DataFrame(columns=['Age#', 'Score (%)', 'city!!'])
    assert clean_names(df).columns.tolist() == ['age', 'score', 'city']


def test_clean_names_mixed():
    """Mixed whitespace and punctuation produce clean snake_case names."""
    df = pd.DataFrame(columns=['First Name', 'Last  Name!', 'Age#'])
    assert clean_names(df).columns.tolist() == ['first_name', 'last_name', 'age']


def test_clean_names_already_clean():
    """Already-clean names are returned unchanged (idempotent)."""
    df = pd.DataFrame(columns=['first_name', 'age', 'score'])
    assert clean_names(df).columns.tolist() == ['first_name', 'age', 'score']


def test_clean_names_upper_case():
    """case='upper' converts all names to uppercase."""
    df = pd.DataFrame(columns=['first name', 'age'])
    assert clean_names(df, case='upper').columns.tolist() == ['FIRST_NAME', 'AGE']


def test_clean_names_snake_alias():
    """case='snake' is an alias for case='lower'."""
    df = pd.DataFrame(columns=['First Name', 'AGE'])
    assert clean_names(df, case='lower').columns.tolist() == \
           clean_names(df, case='snake').columns.tolist()


def test_clean_names_duplicates():
    """Duplicate cleaned names get _2, _3 suffixes."""
    df = pd.DataFrame(columns=['name', 'Name', 'NAME'])
    assert clean_names(df).columns.tolist() == ['name', 'name_2', 'name_3']


def test_clean_names_duplicates_from_cleaning():
    """Names that become identical after cleaning are deduplicated."""
    df = pd.DataFrame(columns=['first name', 'first-name', 'first  name'])
    cols = clean_names(df).columns.tolist()
    assert len(set(cols)) == 3
    assert cols[0] == 'first_name'


def test_clean_names_persian():
    """Persian letters are preserved; spaces become underscores."""
    df = pd.DataFrame(columns=['درآمد ماهانه', 'سن'])
    assert clean_names(df).columns.tolist() == ['درآمد_ماهانه', 'سن']


def test_clean_names_persian_mixed():
    """Mixed Persian and Latin columns are both cleaned correctly."""
    df = pd.DataFrame(columns=['سن (Year)', 'First Name', 'درآمد'])
    cols = clean_names(df).columns.tolist()
    assert 'first_name' in cols
    assert 'درآمد' in cols


def test_clean_names_numeric_start():
    """Names starting with a digit get a leading underscore."""
    df = pd.DataFrame(columns=['1st_place', '2nd'])
    for col in clean_names(df).columns:
        assert not col[0].isdigit()


def test_clean_names_does_not_modify_data():
    """clean_names must not alter the DataFrame's values."""
    df = pd.DataFrame({'First Name': [1, 2], 'Age': [30, 40]})
    result = clean_names(df)
    assert list(result['first_name']) == [1, 2]
    assert list(result['age']) == [30, 40]


def test_clean_names_invalid_case():
    """An invalid case argument raises ValueError."""
    df = pd.DataFrame(columns=['a'])
    with pytest.raises(ValueError, match="case must be one of"):
        clean_names(df, case='title')


def test_clean_names_invalid_sep():
    """A word-character sep raises ValueError."""
    df = pd.DataFrame(columns=['a'])
    with pytest.raises(ValueError, match="sep must be"):
        clean_names(df, sep='a')


# ---------------------------------------------------------------------------
# remove_empty
# ---------------------------------------------------------------------------

def test_remove_empty_drops_empty_col(nan_df):
    """remove_empty drops a fully-NaN column by default."""
    result = remove_empty(nan_df)
    assert 'B' not in result.columns
    assert 'A' in result.columns
    assert 'C' in result.columns


def test_remove_empty_drops_empty_row():
    """remove_empty drops a fully-NaN row by default."""
    df = pd.DataFrame({
        'A': [1.0, np.nan],
        'B': [2.0, np.nan],
    })
    result = remove_empty(df, axis='rows')
    assert len(result) == 1


def test_remove_empty_axis_cols_only(nan_df):
    """axis='cols' drops columns but not rows."""
    result = remove_empty(nan_df, axis='cols')
    assert 'B' not in result.columns
    assert len(result) == len(nan_df)


def test_remove_empty_axis_rows_only(nan_df):
    """axis='rows' drops rows but not columns."""
    result = remove_empty(nan_df, axis='rows')
    assert 'B' in result.columns  # col not dropped


def test_remove_empty_thresh_col():
    """thresh_col drops columns exceeding the missingness fraction."""
    df = pd.DataFrame({
        'A': [1.0, np.nan, 3.0, 4.0],  # 25% missing
        'B': [np.nan, np.nan, np.nan, 4.0],  # 75% missing
    })
    result = remove_empty(df, axis='cols', thresh_col=0.5)
    assert 'A' in result.columns
    assert 'B' not in result.columns


def test_remove_empty_thresh_row():
    """thresh_row drops rows exceeding the missingness fraction."""
    df = pd.DataFrame({
        'A': [1.0, np.nan],
        'B': [2.0, np.nan],
        'C': [3.0, np.nan],
        'D': [4.0, 4.0],
    })
    # row 1 has 75 % missing — should be dropped at thresh 0.5
    result = remove_empty(df, axis='rows', thresh_row=0.5)
    assert len(result) == 1


def test_remove_empty_sentinel():
    """Sentinel values are treated as missing."""
    df = pd.DataFrame({'A': [-99, -99], 'B': [1, 2]})
    result = remove_empty(df, axis='cols', missing_values=[-99])
    assert 'A' not in result.columns
    assert 'B' in result.columns


def test_remove_empty_invalid_axis():
    """Invalid axis raises ValueError."""
    df = pd.DataFrame({'A': [1]})
    with pytest.raises(ValueError, match="axis must be"):
        remove_empty(df, axis='diagonal')


def test_remove_empty_does_not_mutate(nan_df):
    """remove_empty must not mutate the original DataFrame."""
    original_cols = list(nan_df.columns)
    remove_empty(nan_df)
    assert list(nan_df.columns) == original_cols


# ---------------------------------------------------------------------------
# coalesce_columns
# ---------------------------------------------------------------------------

def test_coalesce_basic():
    """First non-null value wins across target and donors."""
    df = pd.DataFrame({
        'a': [1.0, np.nan, np.nan],
        'b': [np.nan, 2.0, np.nan],
        'c': [np.nan, np.nan, 3.0],
    })
    result = coalesce_columns(df, 'a', 'b', 'c')
    assert list(result['a']) == [1.0, 2.0, 3.0]


def test_coalesce_target_priority():
    """Existing non-null values in target are not overwritten."""
    df = pd.DataFrame({'a': [10.0, np.nan], 'b': [99.0, 2.0]})
    result = coalesce_columns(df, 'a', 'b')
    assert result.loc[0, 'a'] == 10.0
    assert result.loc[1, 'a'] == 2.0


def test_coalesce_remove_donors():
    """remove_donors=True drops donor columns from the result."""
    df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
    result = coalesce_columns(df, 'a', 'b', remove_donors=True)
    assert 'b' not in result.columns
    assert 'a' in result.columns


def test_coalesce_all_nan():
    """Rows where all sources are NaN remain NaN."""
    df = pd.DataFrame({'a': [np.nan], 'b': [np.nan]})
    result = coalesce_columns(df, 'a', 'b')
    assert pd.isnull(result.loc[0, 'a'])


def test_coalesce_no_donors_raises():
    """Calling without donors raises ValueError."""
    df = pd.DataFrame({'a': [1.0]})
    with pytest.raises(ValueError, match="donor"):
        coalesce_columns(df, 'a')


def test_coalesce_missing_col_raises():
    """Missing column names raise KeyError."""
    df = pd.DataFrame({'a': [1.0]})
    with pytest.raises(KeyError):
        coalesce_columns(df, 'a', 'nonexistent')


def test_coalesce_does_not_mutate():
    """coalesce_columns must not mutate the original DataFrame.

    Uses pd.array_equiv for comparison because ``nan != nan`` in Python,
    which causes a plain list equality check to fail spuriously.
    """
    df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
    original_a = df['a'].copy()
    coalesce_columns(df, 'a', 'b')
    assert pd.array_equiv(df['a'].values, original_a.values)


# ---------------------------------------------------------------------------
# miss_as_feature
# ---------------------------------------------------------------------------

def test_miss_as_feature_columns_created():
    """Indicator columns are created for columns with missing data."""
    df = pd.DataFrame({'A': [1.0, np.nan], 'B': [1.0, 2.0]})
    result = miss_as_feature(df)
    assert 'A_NA' in result.columns
    assert 'B_NA' not in result.columns  # B has no missing


def test_miss_as_feature_values():
    """Indicator is 1 where missing and 0 where observed."""
    df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
    result = miss_as_feature(df)
    assert list(result['A_NA']) == [0, 1, 0]


def test_miss_as_feature_custom_columns():
    """When columns= is specified, only those columns get indicators."""
    df = pd.DataFrame({'A': [1.0, np.nan], 'B': [np.nan, 2.0]})
    result = miss_as_feature(df, columns=['A'])
    assert 'A_NA' in result.columns
    assert 'B_NA' not in result.columns


def test_miss_as_feature_keep_original_false():
    """keep_original=False drops original columns."""
    df = pd.DataFrame({'A': [1.0, np.nan], 'B': [1.0, 2.0]})
    result = miss_as_feature(df, keep_original=False)
    assert 'A' not in result.columns
    assert 'A_NA' in result.columns


def test_miss_as_feature_custom_suffix():
    """Custom suffix is appended to the indicator column name."""
    df = pd.DataFrame({'A': [1.0, np.nan]})
    result = miss_as_feature(df, suffix='_miss')
    assert 'A_miss' in result.columns


def test_miss_as_feature_sentinel():
    """Sentinel values are treated as missing in the indicator."""
    df = pd.DataFrame({'A': [-99, 1, 2]})
    result = miss_as_feature(df, missing_values=[-99])
    assert 'A_NA' in result.columns
    assert result.loc[0, 'A_NA'] == 1
    assert result.loc[1, 'A_NA'] == 0


def test_miss_as_feature_order():
    """Indicator columns are placed immediately after their source columns."""
    df = pd.DataFrame({'A': [1.0, np.nan], 'B': [np.nan, 2.0], 'C': [1.0, 2.0]})
    result = miss_as_feature(df)
    cols = result.columns.tolist()
    assert cols.index('A_NA') == cols.index('A') + 1
    assert cols.index('B_NA') == cols.index('B') + 1


def test_miss_as_feature_persian():
    """Persian column names are handled correctly in indicator names."""
    df = pd.DataFrame({'درآمد': [1000, np.nan, 3000]})
    result = miss_as_feature(df)
    assert 'درآمد_NA' in result.columns


def test_miss_as_feature_missing_col_raises():
    """Specifying a non-existent column raises KeyError."""
    df = pd.DataFrame({'A': [1.0]})
    with pytest.raises(KeyError):
        miss_as_feature(df, columns=['nonexistent'])


def test_miss_as_feature_no_missing():
    """A fully-observed DataFrame produces no indicator columns."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = miss_as_feature(df)
    assert result.columns.tolist() == ['A', 'B']
