"""Data manipulation utilities for missing data workflows.

This module provides helpers for replacing sentinel values with ``NaN``
and for cleaning DataFrame column names so that downstream code can use
consistent, predictable identifiers.

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
