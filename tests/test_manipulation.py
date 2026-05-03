"""Tests for missingly.manipulation module."""

import pandas as pd
import numpy as np
import pytest

from missingly import manipulation
from missingly.manipulation import clean_names


@pytest.fixture
def sample_df():
    """Numeric and string dataframe with sentinel missing values."""
    return pd.DataFrame({
        'A': [1, -99, 3, 4],
        'B': ['x', 'y', 'N/A', 'z'],
        'C': [10.0, 20.0, 30.0, -99.0],
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
# clean_names — basic ASCII
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


# ---------------------------------------------------------------------------
# clean_names — duplicates
# ---------------------------------------------------------------------------

def test_clean_names_duplicates():
    """Duplicate cleaned names get _2, _3 suffixes."""
    df = pd.DataFrame(columns=['name', 'Name', 'NAME'])
    assert clean_names(df).columns.tolist() == ['name', 'name_2', 'name_3']


def test_clean_names_duplicates_from_cleaning():
    """Names that become identical after cleaning are deduplicated."""
    df = pd.DataFrame(columns=['first name', 'first-name', 'first  name'])
    cols = clean_names(df).columns.tolist()
    assert len(set(cols)) == 3, f"Expected 3 unique names, got: {cols}"
    assert cols[0] == 'first_name'


# ---------------------------------------------------------------------------
# clean_names — Persian / Unicode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# clean_names — edge cases
# ---------------------------------------------------------------------------

def test_clean_names_numeric_start():
    """Names starting with a digit get a leading underscore."""
    df = pd.DataFrame(columns=['1st_place', '2nd'])
    for col in clean_names(df).columns:
        assert not col[0].isdigit(), f"Column {col!r} starts with a digit"


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
