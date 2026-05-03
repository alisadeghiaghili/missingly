"""Data manipulation utilities for missing data workflows.

This module provides helpers for replacing sentinel values with ``NaN``
and for cleaning DataFrame column names so that downstream code can use
consistent, predictable identifiers.
"""

import re
import unicodedata

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Callable


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
        The dataframe itself is not modified; a renamed copy is returned.
    case : {"lower", "upper", "snake"}, optional
        Case transformation to apply.  ``"snake"`` is an alias for
        ``"lower"`` (included for R-janitor familiarity).
        Default is ``"lower"``.
    sep : str, optional
        Separator character inserted between words.  Must be non-empty
        and must not consist solely of alphanumeric characters (letters
        or digits).  ``"_"``, ``"-"``, ``"."`` are all valid.
        Default is ``"_"``.
    strip_accents : bool, optional
        If ``True``, decompose Unicode characters (NFD normalisation)
        and remove combining accent marks (category ``Mn``) before
        processing.  Useful for Latin-script names with diacritics
        (e.g. ``"résumé"`` → ``"resume"``).
        Persian/Arabic letters are *not* decomposed by NFD, so setting
        this to ``True`` is safe for mixed Persian-Latin column names.
        Default is ``False``.

    Returns
    -------
    pd.DataFrame
        A shallow copy of *df* with cleaned column names.  The data is
        not copied.

    Raises
    ------
    ValueError
        If *case* is not one of ``"lower"``, ``"upper"``, ``"snake"``.
    ValueError
        If *sep* is empty or consists solely of alphanumeric characters,
        which would make it indistinguishable from regular word content.

    Notes
    -----
    The function intentionally preserves Persian, Arabic, and CJK
    letters because ``\\w`` in Python's ``re`` module matches all
    Unicode word characters.  If you want pure ASCII column names,
    combine ``strip_accents=True`` with a manual
    ``encode('ascii', 'ignore')`` on the column names beforehand.

    ``"_"`` is a valid separator even though it is technically a
    ``\\w`` character — it is the conventional snake_case separator and
    is treated as a special case in the validation logic.

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
    # sep must be non-empty and must not be purely alphanumeric.
    # '_' is explicitly allowed as the canonical snake_case separator.
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

        # Replace every non-word character with sep.
        # \W matches anything that is not [a-zA-Z0-9_] (Unicode-aware).
        s = re.sub(r"\W+", sep, s, flags=re.UNICODE)

        # Collapse consecutive seps and strip edge seps.
        escaped = re.escape(sep)
        s = re.sub(escaped + "+", sep, s)
        s = s.strip(sep)

        # Python identifier safety: names starting with a digit.
        if s and s[0].isdigit():
            s = sep + s

        return s or sep  # fallback for names that become empty

    raw_names = [_clean_one(col) for col in df.columns]

    # Resolve duplicates: second occurrence → name_2, third → name_3, …
    seen: dict[str, int] = {}
    clean: list[str] = []
    for name in raw_names:
        if name not in seen:
            seen[name] = 1
            clean.append(name)
        else:
            seen[name] += 1
            clean.append(f"{name}{sep}{seen[name]}")

    return df.rename(columns=dict(zip(df.columns, clean)))
