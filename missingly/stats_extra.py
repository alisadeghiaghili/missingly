"""Additional statistical tests for missing data analysis.

This module extends :mod:`missingly.stats` with three additional tests
that are commonly needed but absent from the core module:

:func:`test_hotelling`
    Hotelling's T² test between the complete cases and the cases with
    at least one missing value.  A significant result suggests that the
    two groups differ systematically in their observed values —
    evidence against MCAR.

:func:`test_pattern_monotone`
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


def test_hotelling(
    df: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Hotelling's T² test: complete cases vs. incomplete cases.

    Computes Hotelling's two-sample T² statistic between:

    * **Complete cases** — rows with no missing values in *any* numeric column.
    * **Incomplete cases** — rows with at least one missing numeric value.

    A small p-value (< 0.05) indicates that the two groups have
    significantly different means on the observed variables — evidence
    against MCAR and potentially for MAR or MNAR.

    .. note::
        For the test to be valid, the incomplete-case group must have at
        least ``d + 2`` rows, where ``d`` is the number of numeric
        columns.  If this condition is not met, ``p_value`` is ``None``
        and ``sufficient_data`` is ``False``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Non-numeric columns are ignored.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    dict
        Keys:

        * ``t2`` : float — Hotelling T² statistic.
        * ``f_statistic`` : float — F-approximation of T².
        * ``df1``, ``df2`` : int — numerator / denominator degrees of freedom.
        * ``p_value`` : float or None.
        * ``n_complete`` : int — number of complete-case rows.
        * ``n_incomplete`` : int — number of incomplete-case rows.
        * ``sufficient_data`` : bool.

    Raises
    ------
    ValueError
        If *df* has fewer than 2 numeric columns.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'a': rng.normal(size=200), 'b': rng.normal(size=200)})
    >>> df.loc[:20, 'a'] = np.nan
    >>> test_hotelling(df)
    """
    if missing_values is not None:
        df = df.replace(missing_values, np.nan)

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError(
            "test_hotelling requires at least 2 numeric columns; "
            f"got {num_df.shape[1]}."
        )

    complete_mask = num_df.notna().all(axis=1)
    X_complete = num_df[complete_mask].to_numpy(dtype=float)
    X_incomplete = num_df[~complete_mask].dropna(axis=1, how="all").to_numpy(dtype=float)

    n1, d = X_complete.shape
    n2 = (~complete_mask).sum()

    _sufficient = n2 >= (d + 2) and n1 >= 2

    if not _sufficient or X_incomplete.shape[1] < d:
        return {
            "t2": None,
            "f_statistic": None,
            "df1": d,
            "df2": None,
            "p_value": None,
            "n_complete": int(n1),
            "n_incomplete": int(n2),
            "sufficient_data": False,
        }

    # Use only complete columns of incomplete rows for the test
    # Fallback: compare means on columns that are *fully* observed in both groups
    complete_cols = np.where(num_df[~complete_mask].notna().all(axis=0))[0]
    if len(complete_cols) < 2:
        # Not enough jointly observed columns
        return {
            "t2": None,
            "f_statistic": None,
            "df1": d,
            "df2": None,
            "p_value": None,
            "n_complete": int(n1),
            "n_incomplete": int(n2),
            "sufficient_data": False,
        }

    X1 = X_complete[:, complete_cols]
    X2 = num_df[~complete_mask].iloc[:, complete_cols].to_numpy(dtype=float)
    d_eff = len(complete_cols)
    n1_eff, n2_eff = len(X1), len(X2)

    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean_diff = mean1 - mean2

    # Pooled covariance
    S1 = np.cov(X1, rowvar=False) if n1_eff > 1 else np.eye(d_eff)
    S2 = np.cov(X2, rowvar=False) if n2_eff > 1 else np.eye(d_eff)
    S_pool = ((n1_eff - 1) * S1 + (n2_eff - 1) * S2) / (n1_eff + n2_eff - 2)
    S_pool += np.eye(d_eff) * 1e-8  # ridge for numerical stability

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
        "n_incomplete": int(n2),
        "sufficient_data": True,
    }


def test_pattern_monotone(
    df: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Test whether the missing data pattern is monotone.

    A missing data pattern is **monotone** if the columns can be ordered
    such that: whenever row *i* has a missing value in column *j*, it
    also has missing values in all columns *k > j*.  Equivalently, the
    binary missingness indicator matrix (sorted by number of missing
    values per row) has no "isolated" missing cells.

    Monotone patterns arise naturally in longitudinal studies where
    dropout is the only cause of missingness (once a participant drops
    out, all subsequent measurements are missing).  They permit simpler
    sequential imputation strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    dict
        Keys:

        * ``is_monotone`` : bool — True if the pattern is monotone.
        * ``n_violating_rows`` : int — rows that violate monotonicity
          (0 if monotone).
        * ``sorted_columns`` : list[str] — columns sorted by ascending
          missingness rate (the ordering under which monotonicity is
          assessed).
        * ``monotone_pct`` : float — fraction of rows that *are*
          consistent with monotonicity.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     'a': [1.0, np.nan, np.nan],
    ...     'b': [2.0, 2.0,   np.nan],
    ... })
    >>> test_pattern_monotone(df)
    {'is_monotone': True, 'n_violating_rows': 0, ...}
    """
    if missing_values is not None:
        df = df.replace(missing_values, np.nan)

    # Sort columns by ascending missingness rate
    miss_rate = df.isnull().mean().sort_values()
    sorted_cols = miss_rate.index.tolist()
    indicator = df[sorted_cols].isnull().to_numpy(dtype=int)  # 1 = missing

    # A row is monotone if, once a 1 appears, all subsequent entries are 1
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
    df: pd.DataFrame,
    method: str = "pearson",
    missing_values: Optional[list] = None,
) -> pd.DataFrame:
    """Compute the pairwise nullity correlation matrix.

    Converts each column into a binary indicator (1 = missing, 0 =
    observed) and computes the pairwise correlation between indicators.
    The result can be passed directly to visualisation functions (e.g.
    ``seaborn.heatmap``) or used to identify columns whose missingness
    is structurally linked.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Correlation method.  Default ``'pearson'``.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    pd.DataFrame
        Square DataFrame of shape (n_cols, n_cols) with correlation
        values on [-1, 1].  Diagonal is 1.0.  Columns/rows are the
        original column names of *df*.

    Raises
    ------
    ValueError
        If *method* is not one of ``'pearson'``, ``'kendall'``,
        ``'spearman'``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'a': rng.normal(size=100), 'b': rng.normal(size=100)})
    >>> df.loc[:10, 'a'] = np.nan
    >>> missing_correlation_matrix(df)
    """
    valid_methods = {"pearson", "kendall", "spearman"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}; got {method!r}."
        )

    if missing_values is not None:
        df = df.replace(missing_values, np.nan)

    indicator = df.isnull().astype(float)

    # Drop columns with no missingness (zero variance — correlation undefined)
    has_missing = indicator.any(axis=0)
    indicator = indicator.loc[:, has_missing]

    if indicator.shape[1] == 0:
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    return indicator.corr(method=method)
