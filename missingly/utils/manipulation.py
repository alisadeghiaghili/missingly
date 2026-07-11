"""DataFrame manipulation helpers for missing-data workflows.

This module is the **canonical** home for manipulation utilities that
are in scope for ``missingly`` (replacing sentinels, building shadow
matrices, adding missingness indicator columns).  Functions that fall
outside the missingness charter (``clean_names``, ``remove_empty``,
``coalesce_columns``) remain as deprecated shims pointing to
``data_quality_toolkit``.

Compatibility
-------------
Compatible with Python 3.9+.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from missingly._deprecation import deprecated_api


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
    condition : callable

    Returns
    -------
    pd.DataFrame

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
            "Pass a different col_name to avoid overwriting."
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
        Shape ``(n_rows, n_cols)`` boolean DataFrame, columns ``<col>_NA``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, 3.0]})
    >>> bind_shadow_matrix(df)
        a_NA   b_NA
    0  False   True
    1   True  False
    2  False  False
    """
    shadow = df.isnull()
    if missing_values is not None:
        shadow = shadow | df.isin(missing_values)
    shadow.columns = pd.Index([f"{col}_NA" for col in df.columns])
    return shadow


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import clean_names` instead.",
    since="0.2.0",
)
def clean_names(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    from data_quality_toolkit.cleaning import clean_names as _clean
    return _clean(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import remove_empty` instead.",
    since="0.2.0",
)
def remove_empty(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    from data_quality_toolkit.cleaning import remove_empty as _remove
    return _remove(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.cleaning import coalesce_columns` instead.",
    since="0.2.0",
)
def coalesce_columns(*args, **kwargs):
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.cleaning``.
    """
    from data_quality_toolkit.cleaning import coalesce_columns as _coal
    return _coal(*args, **kwargs)


@deprecated_api(
    "This function may be moved to a separate feature-engineering package.",
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
