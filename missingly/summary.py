"""Summary statistics for missing data analysis.

Provides functions that quantify missingness at the dataset, variable,
and case level.  All functions are pure: they never modify the input
DataFrame and return new objects.

Compatibility
-------------
Compatible with Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import pandas as pd
import numpy as np


def bind_shadow(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Bind a shadow matrix to a DataFrame.

    The shadow matrix contains one boolean column per original column,
    named ``<col>_NA``, where ``True`` indicates a missing value.
    The result has twice as many columns as the input.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to bind the shadow matrix to.
    missing_values : list, optional
        Additional sentinel values treated as missing (NaN always detected).

    Returns
    -------
    pd.DataFrame
        Shape ``(n_rows, 2 * n_cols)``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'A': [1.0, np.nan, -99.0], 'B': [np.nan, 2.0, 3.0]})
    >>> bind_shadow(df, missing_values=[-99])
         A    B   A_NA   B_NA
    0  1.0  NaN  False   True
    1  NaN  2.0   True  False
    2 -99.0  3.0   True  False
    """
    shadow_df = df.isnull()
    if missing_values is not None:
        shadow_df = shadow_df | df.isin(missing_values)
    shadow_df.columns = [f"{col}_NA" for col in df.columns]
    return pd.concat([df, shadow_df], axis=1)


def n_miss(df: pd.DataFrame, missing_values: Optional[List] = None) -> int:
    """Count total missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
        Sentinel values treated as missing in addition to NaN.

    Returns
    -------
    int
    """
    if missing_values is None:
        return int(df.isnull().sum().sum())
    return int((df.isnull() | df.isin(missing_values)).sum().sum())


def n_complete(df: pd.DataFrame, missing_values: Optional[List] = None) -> int:
    """Count total non-missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    int
    """
    return df.size - n_miss(df, missing_values)


def pct_miss(df: pd.DataFrame, missing_values: Optional[List] = None) -> float:
    """Percentage of missing values (0–100).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    float
    """
    return (n_miss(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def pct_complete(df: pd.DataFrame, missing_values: Optional[List] = None) -> float:
    """Percentage of non-missing values (0–100).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    float
    """
    return (n_complete(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def miss_var_summary(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Summarise missingness by variable (column).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``variable``, ``n_miss``, ``pct_miss``.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum()
    miss_pct = (miss_counts / len(df)) * 100 if len(df) > 0 else 0
    return pd.DataFrame({
        'variable': df.columns,
        'n_miss': miss_counts.values,
        'pct_miss': miss_pct.values,
    }).reset_index(drop=True)


def miss_case_summary(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Summarise missingness by case (row).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``case``, ``n_miss``, ``pct_miss``.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum(axis=1)
    miss_pct = (miss_counts / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    return pd.DataFrame({
        'case': df.index,
        'n_miss': miss_counts.values,
        'pct_miss': miss_pct.values,
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# New functions
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """High-level summary of missing data for the whole DataFrame.

    Returns a single-row DataFrame containing dataset-level missingness
    statistics: total cells, missing count, complete count, and
    percentages.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarise.
    missing_values : list, optional
        Sentinel values treated as missing in addition to NaN.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns:
        ``n_rows``, ``n_cols``, ``n_cells``,
        ``n_miss``, ``n_complete``, ``pct_miss``, ``pct_complete``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, 3.0]})
    >>> summarise(df)
       n_rows  n_cols  n_cells  n_miss  n_complete   pct_miss  pct_complete
    0       3       2        6       2           4  33.333333     66.666667
    """
    nm = n_miss(df, missing_values)
    nc = n_complete(df, missing_values)
    total = df.size
    return pd.DataFrame([{
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_cells': total,
        'n_miss': nm,
        'n_complete': nc,
        'pct_miss': (nm / total * 100) if total > 0 else 0.0,
        'pct_complete': (nc / total * 100) if total > 0 else 0.0,
    }])


def miss_scan_count(
    df: pd.DataFrame,
    search: Union[List[Any], Any],
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Count occurrences of specific values (sentinels) per variable.

    Scans each column for values listed in *search* and returns how many
    times each value appears per column.  NaN is matched using
    ``pd.isna`` when ``float('nan')`` or ``None`` is included in *search*.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to scan.
    search : scalar or list
        Value(s) to search for (e.g. ``[-99, -98, "N/A"]``).
    missing_values : list, optional
        Additional background sentinels to treat as missing when
        computing the ``n_miss`` column in the result.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``variable``, ``value``, ``n``.
        One row per (variable, search-value) combination where ``n > 0``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [-99, 1, -99], 'b': [2, -99, 3]})
    >>> miss_scan_count(df, search=[-99])
      variable  value  n
    0        a    -99  2
    1        b    -99  1
    """
    if not isinstance(search, list):
        search = [search]

    rows = []
    for col in df.columns:
        series = df[col]
        for val in search:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                count = int(series.isna().sum())
            else:
                try:
                    count = int((series == val).sum())
                except TypeError:
                    count = 0
            if count > 0:
                rows.append({'variable': col, 'value': val, 'n': count})

    if rows:
        return pd.DataFrame(rows).reset_index(drop=True)
    # Return empty frame with correct schema
    return pd.DataFrame(columns=['variable', 'value', 'n'])


def miss_summary(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    digits: int = 2,
) -> dict:
    """Comprehensive missing-data summary combining variable and case views.

    Convenience wrapper that calls :func:`summarise`, :func:`miss_var_summary`,
    and :func:`miss_case_summary` and returns all three results as a dict.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarise.
    missing_values : list, optional
        Sentinel values treated as missing in addition to NaN.
    digits : int, default 2
        Number of decimal places to round ``pct_miss`` columns.

    Returns
    -------
    dict with keys:
        ``"summary"``      — one-row DataFrame from :func:`summarise`.
        ``"by_variable"``  — per-column DataFrame from :func:`miss_var_summary`.
        ``"by_case"``      — per-row DataFrame from :func:`miss_case_summary`.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
    >>> result = miss_summary(df)
    >>> list(result.keys())
    ['summary', 'by_variable', 'by_case']
    """
    def _round(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if 'pct_miss' in out.columns:
            out['pct_miss'] = out['pct_miss'].round(digits)
        if 'pct_complete' in out.columns:
            out['pct_complete'] = out['pct_complete'].round(digits)
        return out

    return {
        'summary': _round(summarise(df, missing_values)),
        'by_variable': _round(miss_var_summary(df, missing_values)),
        'by_case': _round(miss_case_summary(df, missing_values)),
    }
