"""Data manipulation utilities (moved from missingly root in v0.3.0).

This module contains the canonical implementations of data-cleaning
helpers that are kept inside ``missingly`` solely for backward
compatibility.  They are **out of scope** for the missingness-analysis
charter; new user code should use ``data_quality_toolkit`` instead.

.. deprecated:: 0.3.0
    Import directly from ``missingly.manipulation`` (which re-exports
    these symbols with a :class:`FutureWarning`) or from
    ``data_quality_toolkit.cleaning`` (preferred).

Removal target: v0.4.0.
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

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1, -99, 3], 'b': ['x', 'N/A', 'z']})
    >>> replace_with_na(df, replace={'a': -99, 'b': 'N/A'})
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
    """Replace all values in a DataFrame with ``NaN`` if they meet a condition.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    condition : callable
        A function that accepts a single cell value and returns ``True``
        if that cell should be replaced with ``NaN``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with matching values replaced by ``NaN``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, -99, 3], 'b': [-99, 2, -99]})
    >>> replace_with_na_all(df, condition=lambda x: x == -99)
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
    """Add a boolean column indicating whether each row has any missing value.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    col_name : str, default ``"any_miss"``

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If *col_name* already exists in *df*.
    """
    if col_name in df.columns:
        raise ValueError(
            f"Column {col_name!r} already exists in the DataFrame. "
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
    """Return the shadow matrix of a DataFrame as a standalone DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Shape ``(n_rows, n_cols)`` with boolean dtype, column names
        ``["<original_col>_NA", ...]``.
    """
    shadow = df.isnull()
    if missing_values is not None:
        shadow = shadow | df.isin(missing_values)
    shadow.columns = [f"{col}_NA" for col in df.columns]
    return shadow


def miss_as_feature(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    missing_values: Optional[List] = None,
    suffix: str = "_NA",
    keep_original: bool = True,
) -> pd.DataFrame:
    """Encode missingness as binary indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
    missing_values : list, optional
    suffix : str, default ``"_NA"``
    keep_original : bool, default True

    Returns
    -------
    pd.DataFrame
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
