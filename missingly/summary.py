"""Summary statistics for missing data analysis.

Provides functions that quantify missingness at the dataset, variable,
and case level.  All functions are pure: they never modify the input
DataFrame and return new objects.

Compatibility
-------------
Compatible with Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np


def bind_shadow(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Bind a shadow matrix to a DataFrame.

    The shadow matrix contains one boolean column per original column,
    named ``<col>_NA``, where ``True`` indicates a missing value.
    The result has twice as many columns as the input.

    This implements the full naniar shadow-matrix contract:
    - Standard ``NaN`` / ``None`` values are **always** detected.
    - When *missing_values* is provided, those sentinel values are
      **also** treated as missing (in addition to ``NaN``).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to bind the shadow matrix to.
    missing_values : list, optional
        A list of additional sentinel values to be considered as missing.
        Standard ``NaN`` is always detected regardless of this parameter.

    Returns
    -------
    pd.DataFrame
        A new dataframe with the shadow matrix bound to the right.
        Shape is ``(n_rows, 2 * n_cols)``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'A': [1.0, np.nan, -99.0], 'B': [np.nan, 2.0, 3.0]})
    >>> bind_shadow(df, missing_values=[-99])
         A    B   A_NA   B_NA
    0  1.0  NaN  False   True
    1  NaN  2.0   True  False
    2 -99.0  3.0   True  False
    """
    # Always detect real NaN/None
    shadow_df = df.isnull()

    # Overlay sentinel values as missing (OR with isnull mask)
    if missing_values is not None:
        shadow_df = shadow_df | df.isin(missing_values)

    shadow_df.columns = [f"{col}_NA" for col in df.columns]
    return pd.concat([df, shadow_df], axis=1)


def n_miss(df: pd.DataFrame, missing_values: Optional[List] = None) -> int:
    """Count the total number of missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to count missing values in.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    int
        The total number of missing values.
    """
    if missing_values is None:
        return int(df.isnull().sum().sum())
    return int((df.isnull() | df.isin(missing_values)).sum().sum())


def n_complete(df: pd.DataFrame, missing_values: Optional[List] = None) -> int:
    """Count the total number of non-missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to count non-missing values in.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    int
        The total number of non-missing values.
    """
    return df.size - n_miss(df, missing_values)


def pct_miss(df: pd.DataFrame, missing_values: Optional[List] = None) -> float:
    """Calculate the percentage of missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the percentage of missing values in.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    float
        The percentage of missing values (0–100).
    """
    return (n_miss(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def pct_complete(df: pd.DataFrame, missing_values: Optional[List] = None) -> float:
    """Calculate the percentage of non-missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the percentage of non-missing values in.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    float
        The percentage of non-missing values (0–100).
    """
    return (n_complete(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def miss_var_summary(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Summarise missingness by variable (column).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarise.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per variable and columns:
        ``variable``, ``n_miss``, ``pct_miss``.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum()
    miss_pct = (miss_counts / len(df)) * 100 if len(df) > 0 else 0
    summary_df = pd.DataFrame({
        'variable': df.columns,
        'n_miss': miss_counts.values,
        'pct_miss': miss_pct.values,
    })
    return summary_df.reset_index(drop=True)


def miss_case_summary(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Summarise missingness by case (row).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarise.
    missing_values : list, optional
        A list of values to be considered as missing in addition to ``NaN``.

    Returns
    -------
    pd.DataFrame
        A dataframe with one row per case and columns:
        ``case``, ``n_miss``, ``pct_miss``.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum(axis=1)
    miss_pct = (miss_counts / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    summary_df = pd.DataFrame({
        'case': df.index,
        'n_miss': miss_counts.values,
        'pct_miss': miss_pct.values,
    })
    return summary_df.reset_index(drop=True)
