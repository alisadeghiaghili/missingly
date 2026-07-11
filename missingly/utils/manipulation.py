"""Data manipulation utilities for missing-data workflows.

This module is the canonical home for DataFrame manipulation helpers
that support missingness analysis.  It was relocated from
``missingly/manipulation.py`` (root) as part of C3 (API surface cleanup).

The root ``manipulation.py`` is now a deprecated shim that re-exports
everything from here and emits :class:`FutureWarning` on import.

Public functions
----------------
replace_with_na
    Replace specified per-column values with ``NaN``.
replace_with_na_all
    Replace every value matching a predicate with ``NaN``.
add_any_miss_var
    Append a boolean column indicating whether each row has any missing value.
bind_shadow_matrix
    Return the shadow matrix of a DataFrame as a standalone DataFrame.

Note: ``clean_names``, ``remove_empty``, ``coalesce_columns``, and
``miss_as_feature`` are **not** reproduced here.  They were deprecated in
v0.2.0 and their stubs remain in the root ``manipulation.py`` shim until
the v0.4.0 deletion milestone.

Compatibility
-------------
Requires Python 3.9+.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def replace_with_na(
    df: pd.DataFrame,
    replace: Dict[str, Union[List, object, Callable]],
) -> pd.DataFrame:
    """Replace specified values in a DataFrame with ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    replace : dict
        Keys are column names; values describe which entries to replace:

        * scalar — replace that exact value;
        * list of scalars — replace any value in the list;
        * callable — replace where ``callable(cell)`` is ``True``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the specified values replaced by ``NaN``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, -99, 3], 'b': ['x', 'N/A', 'z']})
    >>> replace_with_na(df, {'a': -99, 'b': 'N/A'})
         a     b
    0  1.0     x
    1  NaN  None
    2  3.0     z
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
    """Replace every value matching *condition* with ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    condition : callable
        Accepts a single cell value; returns ``True`` if it should become
        ``NaN``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with matching cells replaced by ``NaN``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, -99, 3], 'b': [-99, 2, -99]})
    >>> replace_with_na_all(df, lambda x: x == -99)
         a    b
    0  1.0  NaN
    1  NaN  2.0
    2  3.0  NaN
    """
    return df.map(lambda x: np.nan if condition(x) else x)


def add_any_miss_var(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    col_name: str = "any_miss",
) -> pd.DataFrame:
    """Append a boolean column indicating whether each row has any missing value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Not modified in place.
    missing_values : list, optional
        Additional sentinel values treated as missing alongside ``NaN``.
    col_name : str, default ``"any_miss"``
        Name of the indicator column to append.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with one extra boolean column appended on the right.

    Raises
    ------
    ValueError
        If *col_name* already exists in *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
    >>> add_any_miss_var(df)
         a    b  any_miss
    0  1.0  4.0     False
    1  NaN  5.0      True
    2  3.0  NaN      True
    """
    if col_name in df.columns:
        raise ValueError(
            f"Column {col_name!r} already exists. "
            f"Pass a different col_name to avoid overwriting."
        )
    result = df.copy()
    miss_mask = df.isnull()
    if missing_values is not None:
        miss_mask = miss_mask | df.isin(missing_values)
    result[col_name] = miss_mask.any(axis=1)
    return result


def bind_shadow_matrix(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Return the shadow matrix of *df* as a standalone boolean DataFrame.

    Each output column corresponds to one input column, renamed
    ``<col>_NA``, and contains ``True`` where the original value is
    missing and ``False`` where it is present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Not modified in place.
    missing_values : list, optional
        Additional sentinel values treated as missing alongside ``NaN``.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_rows, n_cols)`` with boolean dtype and column names
        ``["<col>_NA", ...]``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
    >>> bind_shadow_matrix(df)
        a_NA   b_NA
    0  False   True
    1   True  False
    """
    shadow = df.isnull()
    if missing_values is not None:
        shadow = shadow | df.isin(missing_values)
    shadow.columns = [f"{col}_NA" for col in df.columns]
    return shadow
