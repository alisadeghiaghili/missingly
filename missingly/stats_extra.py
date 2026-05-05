"""Additional statistical tests for missing data analysis.

This module extends :mod:`missingly.stats` with three additional tests
that are commonly needed but absent from the core module:

:func:`hotelling_test`
    Hotelling's T² test between the complete cases and the cases with
    at least one missing value.  A significant result suggests that the
    two groups differ systematically in their observed values —
    evidence against MCAR.

:func:`pattern_monotone_test`
    Tests whether the missing data pattern is monotone (i.e. if a row
    is missing column j, it is also missing all columns to the right).
    Monotone patterns have better theoretical properties and are easier
    to handle with sequential imputation.

:func:`missing_correlation_matrix`
    Returns the full pairwise nullity correlation matrix (Pearson
    correlation of the binary indicator columns ``1 = missing,
    0 = observed``).  Unlike the scalar ``max_nullity_corr`` returned by
    :func:`~missingly.stats.diagnose_missing`, this gives the full
    picture and can be passed directly to visualisation functions.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


def hotelling_test(
    frame: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Hotelling's T² test: complete cases vs. incomplete cases.

    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame.  Non-numeric columns are ignored.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    dict
        Keys: ``t2``, ``f_statistic``, ``df1``, ``df2``, ``p_value``,
        ``n_complete``, ``n_incomplete``, ``sufficient_data``.

    Raises
    ------
    ValueError
        If *frame* has fewer than 2 numeric columns.
    """
    if missing_values is not None:
        frame = frame.replace(missing_values, np.nan)

    num_df = frame.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError(
            "hotelling_test requires at least 2 numeric columns; "
            f"got {num_df.shape[1]}."
        )

    complete_mask = num_df.notna().all(axis=1)
    X_complete = num_df[complete_mask].to_numpy(dtype=float)
    n1, d = X_complete.shape
    n2 = int((~complete_mask).sum())

    _sufficient = n2 >= (d + 2) and n1 >= 2

    if not _sufficient:
        return {
            "t2": None,
            "f_statistic": None,
            "df1": d,
            "df2": None,
            "p_value": None,
            "n_complete": int(n1),
            "n_incomplete": n2,
            "sufficient_data": False,
        }

    complete_cols = np.where(num_df[~complete_mask].notna().all(axis=0))[0]
    if len(complete_cols) < 2:
        return {
            "t2": None,
            "f_statistic": None,
            "df1": d,
            "df2": None,
            "p_value": None,
            "n_complete": int(n1),
            "n_incomplete": n2,
            "sufficient_data": False,
        }

    X1 = X_complete[:, complete_cols]
    X2 = num_df[~complete_mask].iloc[:, complete_cols].to_numpy(dtype=float)
    d_eff = len(complete_cols)
    n1_eff, n2_eff = len(X1), len(X2)

    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean_diff = mean1 - mean2

    S1 = np.cov(X1, rowvar=False) if n1_eff > 1 else np.eye(d_eff)
    S2 = np.cov(X2, rowvar=False) if n2_eff > 1 else np.eye(d_eff)
    S_pool = ((n1_eff - 1) * S1 + (n2_eff - 1) * S2) / (n1_eff + n2_eff - 2)
    S_pool += np.eye(d_eff) * 1e-8

    S_inv = np.linalg.inv(S_pool)
    T2 = (n1_eff * n2_eff / (n1_eff + n2_eff)) * float(
        mean_diff @ S_inv @ mean_diff
    )

    df1 = d_eff
    df2 = n1_eff + n2_eff - d_eff - 1
    factor = (n1_eff + n2_eff - d_eff - 1) / ((n1_eff + n2_eff - 2) * d_eff)
    F = T2 * factor
    p_value = float(1 - f_dist.cdf(F, df1, df2)) if df2 > 0 else None

    return {
        "t2": float(T2),
        "f_statistic": float(F),
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "n_complete": int(n1),
        "n_incomplete": n2,
        "sufficient_data": True,
    }


def pattern_monotone_test(
    frame: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Test whether the missing data pattern is monotone.

    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    dict
        Keys: ``is_monotone``, ``n_violating_rows``, ``sorted_columns``,
        ``monotone_pct``.
    """
    if missing_values is not None:
        frame = frame.replace(missing_values, np.nan)

    miss_rate = frame.isnull().mean().sort_values()
    sorted_cols = miss_rate.index.tolist()
    indicator = frame[sorted_cols].isnull().to_numpy(dtype=int)

    n_rows = len(indicator)
    n_violating = 0
    for row in indicator:
        first_missing = None
        violated = False
        for j, val in enumerate(row):
            if val == 1 and first_missing is None:
                first_missing = j
            if first_missing is not None and val == 0:
                violated = True
                break
        if violated:
            n_violating += 1

    return {
        "is_monotone": n_violating == 0,
        "n_violating_rows": n_violating,
        "sorted_columns": sorted_cols,
        "monotone_pct": float((n_rows - n_violating) / n_rows) if n_rows > 0 else 1.0,
    }


def missing_correlation_matrix(
    frame: pd.DataFrame,
    method: str = "pearson",
    missing_values: Optional[list] = None,
) -> pd.DataFrame:
    """Compute the pairwise nullity correlation matrix.

    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Correlation method.  Default ``'pearson'``.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix over columns that have missing values.
        Empty DataFrame (shape 0×0) when no column has missing values.

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    """
    valid_methods = {"pearson", "kendall", "spearman"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}; got {method!r}."
        )

    if missing_values is not None:
        frame = frame.replace(missing_values, np.nan)

    indicator = frame.isnull().astype(float)
    has_missing = indicator.any(axis=0)
    indicator = indicator.loc[:, has_missing]

    if indicator.shape[1] == 0:
        return pd.DataFrame(dtype=float)

    corr = indicator.corr(method=method)
    corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return corr
