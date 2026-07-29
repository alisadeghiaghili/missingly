"""Data manipulation utilities for missing-data workflows.

This module is the canonical home for DataFrame manipulation helpers
that support missingness analysis.  It was relocated from
``missingly/manipulation.py`` (root) as part of C3 (API surface cleanup).

The root ``manipulation.py`` is now a deprecated shim that re-exports
everything from here and emits :class:`DeprecationWarning` on import.

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

Deprecated shims (emit DeprecationWarning on call)
---------------------------------------------------
clean_names
    Moved to ``data_quality_toolkit.cleaning``.
remove_empty
    Moved to ``data_quality_toolkit.cleaning``.
coalesce_columns
    Moved to ``data_quality_toolkit.cleaning``.
miss_as_feature
    Experimental; may be moved to a separate package.

Compatibility
-------------
Requires Python 3.9+.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from missingly._deprecation import deprecated_api


# ---------------------------------------------------------------------------
# Active public API
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deprecated shims
# ---------------------------------------------------------------------------

@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import clean_names` instead.",
    since="0.2.0",
)
def clean_names(*args, **kwargs):
    """Legacy shim — emits :class:`DeprecationWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "clean_names moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import clean_names as _clean  # type: ignore[import]
    return _clean(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import remove_empty` instead.",
    since="0.2.0",
)
def remove_empty(*args, **kwargs):
    """Legacy shim — emits :class:`DeprecationWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "remove_empty moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import remove_empty as _remove  # type: ignore[import]
    return _remove(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import coalesce_columns` instead.",
    since="0.2.0",
)
def coalesce_columns(*args, **kwargs):
    """Legacy shim — emits :class:`DeprecationWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    warnings.warn(
        "coalesce_columns moved to data_quality_toolkit.cleaning and will be "
        "removed from missingly in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.cleaning import coalesce_columns as _coal  # type: ignore[import]
    return _coal(*args, **kwargs)


@deprecated_api(
    "This function may be moved to a separate feature-engineering package in a future release.",
    since="0.2.0",
)
def miss_as_feature(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    *,
    missing_values: Optional[List] = None,
    suffix: str = "_NA",
    keep_original: bool = True,
) -> pd.DataFrame:
    """Encode missingness as binary indicator columns (experimental).

    .. deprecated:: 0.2.0
        Experimental — may be moved to a separate package.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to encode.  Defaults to all columns with any missing values.
    missing_values : list, optional
        Additional sentinel values to treat as missing.
    suffix : str, default ``"_NA"``
        Suffix appended to each indicator column name.
    keep_original : bool, default True
        If False, drop the original columns after adding indicators.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary indicator columns interleaved.
    """
    warnings.warn(
        "miss_as_feature is experimental and may be moved to a separate "
        "package in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
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
