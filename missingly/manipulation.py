"""Data manipulation utilities for missing data workflows.

This module provides helpers for replacing sentinel values with ``NaN``,
cleaning column names, removing empty rows/columns, coalescing columns,
and encoding missingness as binary features.

Compatibility
-------------
Compatible with Python 3.9+.  Uses ``from __future__ import annotations``
for lazy annotation evaluation and ``typing`` generics instead of the
``X | Y`` union syntax introduced in Python 3.10.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np


def replace_with_na(
    df: pd.DataFrame,
    replace: Dict[str, Union[List, object, Callable]],
) -> pd.DataFrame:
    """Replace specified values in a DataFrame with ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify.
    replace : dict
        A dictionary whose keys are column names and whose values
        describe which entries to replace.  Each value may be:

        * a single scalar — replace that exact value;
        * a list of scalars — replace any value in the list;
        * a callable — replace where ``callable(cell)`` returns ``True``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with the specified values replaced with ``NaN``.
        The original dataframe is not modified.

    Example
    -------
    >>> df = pd.DataFrame({'A': [1, -99, 3], 'B': ['x', 'N/A', 'y']})
    >>> replace_with_na(df, {'A': -99, 'B': 'N/A'})
         A    B
    0  1.0    x
    1  NaN  NaN
    2  3.0    y
    """
    df_copy = df.copy()
    for col, condition in replace.items():
        if callable(condition):
            df_copy.loc[df_copy[col].apply(condition), col] = np.nan
        elif isinstance(condition, list):
            df_copy[col] = df_copy[col].replace(condition, np.nan)
        else:
            df_copy[col] = df_copy[col].replace([condition], np.nan)
    return df_copy


def replace_with_na_all(df: pd.DataFrame, condition: Callable) -> pd.DataFrame:
    """Replace all values in a DataFrame with ``NaN`` if they meet a condition.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify.
    condition : callable
        A function that accepts a single cell value and returns ``True``
        if that value should be replaced with ``NaN``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with matching values replaced with ``NaN``.
        The original dataframe is not modified.

    Example
    -------
    >>> df = pd.DataFrame({'A': [1, -99], 'B': ['ok', 'N/A']})
    >>> replace_with_na_all(df, lambda x: x in (-99, 'N/A'))
         A     B
    0  1.0    ok
    1  NaN  NaN
    """
    return df.map(lambda x: np.nan if condition(x) else x)


def clean_names(
    df: pd.DataFrame,
    *,
    case: str = "lower",
    sep: str = "_",
    strip_accents: bool = False,
) -> pd.DataFrame:
    """Normalise DataFrame column names to clean, consistent identifiers.

    Inspired by ``janitor::clean_names`` from R.

    The transformation pipeline applied to each column name:

    1. Convert to string (handles non-string column labels).
    2. Optionally decompose and strip Unicode accent marks
       (``strip_accents=True``).
    3. Apply case transformation (``lower``, ``upper``, or ``snake``).
    4. Replace any character that is *not* a Unicode word character
       (``\\w``, i.e. letters, digits, underscore — including Persian,
       Arabic, CJK, etc.) with the separator character.
    5. Collapse consecutive separators into one.
    6. Strip leading/trailing separators.
    7. If the name starts with a digit, prepend the separator so the
       result is a valid Python identifier.
    8. Resolve duplicate names by appending ``_2``, ``_3``, … suffixes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose column names should be cleaned.
    case : {"lower", "upper", "snake"}, optional
        Case transformation to apply.  Default is ``"lower"``.
    sep : str, optional
        Separator character.  Must be non-empty and not purely alphanumeric.
        Default is ``"_"``.
    strip_accents : bool, optional
        Strip combining accent marks via NFD decomposition.
        Default is ``False``.

    Returns
    -------
    pd.DataFrame
        A shallow copy of *df* with cleaned column names.

    Raises
    ------
    ValueError
        If *case* is invalid or *sep* is empty / purely alphanumeric.

    Example
    -------
    >>> import pandas as pd
    >>> from missingly.manipulation import clean_names
    >>> df = pd.DataFrame(columns=['First Name', 'Last  Name!', 'Age#'])
    >>> clean_names(df).columns.tolist()
    ['first_name', 'last_name', 'age']

    >>> df2 = pd.DataFrame(columns=['درآمد ماهانه', 'سن (Year)'])
    >>> clean_names(df2).columns.tolist()
    ['درآمد_ماهانه', 'سن_year']
    """
    valid_cases = {"lower", "upper", "snake"}
    if case not in valid_cases:
        raise ValueError(
            f"case must be one of {valid_cases!r}; got {case!r}"
        )
    if not sep or (re.fullmatch(r"[A-Za-z0-9]+", sep) is not None):
        raise ValueError(
            f"sep must be a non-empty, non-alphanumeric string "
            f"(e.g. '_', '-', '.'); got {sep!r}"
        )

    def _clean_one(name: str) -> str:
        """Apply the full normalisation pipeline to a single name string."""
        s = str(name)
        if strip_accents:
            s = unicodedata.normalize("NFD", s)
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        if case in ("lower", "snake"):
            s = s.lower()
        elif case == "upper":
            s = s.upper()
        s = re.sub(r"\W+", sep, s, flags=re.UNICODE)
        escaped = re.escape(sep)
        s = re.sub(escaped + "+", sep, s)
        s = s.strip(sep)
        if s and s[0].isdigit():
            s = sep + s
        return s or sep

    raw_names = [_clean_one(col) for col in df.columns]

    seen: Dict[str, int] = {}
    clean: List[str] = []
    for name in raw_names:
        if name not in seen:
            seen[name] = 1
            clean.append(name)
        else:
            seen[name] += 1
            clean.append(f"{name}{sep}{seen[name]}")

    return df.rename(columns=dict(zip(df.columns, clean)))


def remove_empty(
    df: pd.DataFrame,
    *,
    axis: Union[str, int] = "both",
    missing_values: Optional[List] = None,
    thresh_row: Optional[float] = None,
    thresh_col: Optional[float] = None,
) -> pd.DataFrame:
    """Remove rows and/or columns that are entirely (or mostly) missing.

    Inspired by ``janitor::remove_empty`` from R.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to filter.
    axis : {"rows", "cols", "both", 0, 1}, optional
        Which axis to clean.

        * ``"rows"`` / ``0`` — drop empty rows only.
        * ``"cols"`` / ``1`` — drop empty columns only.
        * ``"both"`` (default) — drop empty rows **and** columns.

    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    thresh_row : float, optional
        Fraction threshold in the range **(0, 1]**.  Rows with a missing
        fraction **strictly greater than** ``thresh_row`` are dropped.  When
        ``None`` (default), only fully-empty rows are dropped.

        .. note::
           A value of ``0`` would drop every row (even rows with no missing
           data), which is almost never the desired behaviour.  Pass ``None``
           to drop only fully-empty rows.
    thresh_col : float, optional
        Fraction threshold in the range **(0, 1]**.  Columns with a missing
        fraction **strictly greater than** ``thresh_col`` are dropped.  When
        ``None`` (default), only fully-empty columns are dropped.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the qualifying rows/columns removed.
        The original dataframe is not modified.

    Raises
    ------
    ValueError
        If *axis* is not one of the recognised values.
    ValueError
        If *thresh_row* or *thresh_col* is not in (0, 1].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly.manipulation import remove_empty
    >>> df = pd.DataFrame({
    ...     'A': [1, np.nan, 3],
    ...     'B': [np.nan, np.nan, np.nan],
    ...     'C': [1, np.nan, 3],
    ... })
    >>> remove_empty(df)  # drops column B only
         A    C
    0  1.0  1.0
    1  NaN  NaN
    2  3.0  3.0
    """
    valid_axes = {"rows", "cols", "both", 0, 1}
    if axis not in valid_axes:
        raise ValueError(
            f"axis must be one of {valid_axes!r}; got {axis!r}"
        )
    for name, thresh in (("thresh_row", thresh_row), ("thresh_col", thresh_col)):
        if thresh is not None and not (0.0 < thresh <= 1.0):
            raise ValueError(
                f"{name} must be in the range (0, 1]; got {thresh!r}. "
                f"Pass None to drop only fully-empty rows/columns."
            )

    def _is_missing(frame: pd.DataFrame) -> pd.DataFrame:
        """Boolean mask of missing cells, including sentinels."""
        if missing_values is None:
            return frame.isnull()
        return frame.isnull() | frame.isin(missing_values)

    result = df.copy()
    n_rows, n_cols = result.shape

    drop_rows = axis in ("rows", "both", 0)
    drop_cols = axis in ("cols", "both", 1)

    if drop_cols and n_cols > 0:
        miss_frac_col = _is_missing(result).mean(axis=0)
        if thresh_col is None:
            cols_to_drop = miss_frac_col[miss_frac_col == 1.0].index.tolist()
        else:
            cols_to_drop = miss_frac_col[miss_frac_col > thresh_col].index.tolist()
        result = result.drop(columns=cols_to_drop)

    if drop_rows and result.shape[0] > 0:
        miss_frac_row = _is_missing(result).mean(axis=1)
        if thresh_row is None:
            rows_to_drop = miss_frac_row[miss_frac_row == 1.0].index.tolist()
        else:
            rows_to_drop = miss_frac_row[miss_frac_row > thresh_row].index.tolist()
        result = result.drop(index=rows_to_drop)

    return result


def coalesce_columns(
    df: pd.DataFrame,
    target: str,
    *donors: str,
    remove_donors: bool = False,
) -> pd.DataFrame:
    """Fill missing values in a column using donor columns (SQL COALESCE).

    For each row, the first non-null value encountered left-to-right
    across ``target`` then ``donors`` is used.  Rows where all sources
    are ``NaN`` remain ``NaN`` in the result.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to operate on.
    target : str
        Name of the column to fill.  The result is stored back into
        this column.
    *donors : str
        One or more column names to draw fill values from, evaluated
        left-to-right.
    remove_donors : bool, optional
        If ``True``, drop the donor columns from the result.
        Default is ``False``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with missing values in *target* filled from
        *donors*.  The original dataframe is not modified.

    Raises
    ------
    KeyError
        If *target* or any donor column is not present in *df*.
    ValueError
        If no donor columns are provided.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly.manipulation import coalesce_columns
    >>> df = pd.DataFrame({
    ...     'a': [1.0, np.nan, np.nan],
    ...     'b': [np.nan, 2.0, np.nan],
    ...     'c': [np.nan, np.nan, 3.0],
    ... })
    >>> coalesce_columns(df, 'a', 'b', 'c')
         a    b    c
    0  1.0  NaN  NaN
    1  2.0  2.0  NaN
    2  3.0  NaN  3.0
    """
    if not donors:
        raise ValueError("At least one donor column must be provided.")
    missing_cols = [c for c in (target, *donors) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    result = df.copy()
    filled = result[target].copy()
    for donor in donors:
        filled = filled.combine_first(result[donor])
    result[target] = filled

    if remove_donors:
        result = result.drop(columns=list(donors))

    return result


def miss_as_feature(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    missing_values: Optional[List] = None,
    suffix: str = "_NA",
    keep_original: bool = True,
) -> pd.DataFrame:
    """Encode missingness as binary indicator columns.

    For each selected column, a new binary column ``<col><suffix>`` is
    appended: ``1`` where the value is missing, ``0`` otherwise.  This
    turns the *pattern* of missing data into an explicit feature that
    models can learn from — a technique sometimes called
    "missing indicator" encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to operate on.
    columns : list of str, optional
        Columns for which to create indicators.  When ``None`` (default),
        indicators are created for **all** columns that have at least one
        missing value.
    missing_values : list, optional
        Sentinel values treated as missing in addition to ``NaN``.
    suffix : str, optional
        Suffix appended to the original column name to form the indicator
        column name.  Default is ``"_NA"``.
    keep_original : bool, optional
        If ``True`` (default), the original columns are retained alongside
        the new indicator columns.  If ``False``, original columns are
        dropped after the indicators are created.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with new binary indicator columns appended.
        The original dataframe is not modified.

    Raises
    ------
    KeyError
        If any column in *columns* is not present in *df*.

    Notes
    -----
    The indicator columns are placed immediately after their source column
    in the result, preserving the overall column order.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly.manipulation import miss_as_feature
    >>> df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [np.nan, 2.0, 3.0]})
    >>> miss_as_feature(df)
         A  A_NA    B  B_NA
    0  1.0     0  NaN     1
    1  NaN     1  2.0     0
    2  3.0     0  3.0     0
    """
    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        target_cols = columns
    else:
        null_mask = df.isnull()
        if missing_values:
            null_mask = null_mask | df.isin(missing_values)
        target_cols = [c for c in df.columns if null_mask[c].any()]

    result = df.copy()
    new_order: List[str] = []

    for col in df.columns:
        new_order.append(col)
        if col in target_cols:
            indicator = df[col].isnull()
            if missing_values:
                indicator = indicator | df[col].isin(missing_values)
            ind_name = f"{col}{suffix}"
            result[ind_name] = indicator.astype(int)
            new_order.append(ind_name)

    result = result[new_order]

    if not keep_original:
        result = result.drop(columns=list(target_cols))

    return result
