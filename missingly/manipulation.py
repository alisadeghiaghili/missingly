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

import warnings
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
    """Replace all values in a DataFrame with ``NaN`` if they meet a condition."""
    return df.map(lambda x: np.nan if condition(x) else x)


def clean_names(*args, **kwargs):
    """Deprecated shim for :func:`data_quality_toolkit.cleaning.clean_names`.

    This function will be removed in a future release. Use::

        from data_quality_toolkit import clean_names

    instead.
    """
    warnings.warn(
        "clean_names moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import clean_names as _clean
    return _clean(*args, **kwargs)


def remove_empty(*args, **kwargs):
    """Deprecated shim for :func:`data_quality_toolkit.cleaning.remove_empty`.

    Use ``data_quality_toolkit.cleaning.remove_empty`` instead.
    """
    warnings.warn(
        "remove_empty moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import remove_empty as _remove
    return _remove(*args, **kwargs)


def coalesce_columns(*args, **kwargs):
    """Deprecated shim for :func:`data_quality_toolkit.cleaning.coalesce_columns`.

    Use ``data_quality_toolkit.cleaning.coalesce_columns`` instead.
    """
    warnings.warn(
        "coalesce_columns moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import coalesce_columns as _coal
    return _coal(*args, **kwargs)


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
    appended: ``1`` where the value is missing, ``0`` otherwise.
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
