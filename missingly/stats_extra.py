"""Additional statistical tests for missing data analysis.

This module extends :mod:`missingly.stats` with three additional tests
that are commonly needed but absent from the core module:

- hotelling_test (moved to data_quality_toolkit.statistics)
- pattern_monotone_test
- missing_correlation_matrix

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def hotelling_test(
    frame: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Deprecated shim for data_quality_toolkit.statistics.hotelling_test.

    This function was moved to :mod:`data_quality_toolkit.statistics` and
    will be removed from ``missingly`` in a future release. Import and
    use it from there instead.
    """
    import warnings

    warnings.warn(
        "hotelling_test moved to data_quality_toolkit.statistics and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.statistics import hotelling_test as _hotelling
    return _hotelling(frame=frame, missing_values=missing_values)


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


pattern_monotone_test.__test__ = False  # type: ignore[attr-defined]


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
