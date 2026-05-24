"""Additional statistical tests for missing data analysis.

.. deprecated:: 0.2.0
    These functions are experimental / legacy and emit :class:`FutureWarning`.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ._deprecation import deprecated_api

# Prevent pytest from treating decorated functions as test items.
__test__ = False


@deprecated_api(
    "Use `from data_quality_toolkit.statistics import hotelling_test` instead.",
    since="0.2.0",
)
def hotelling_test(
    frame: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Legacy shim — emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.statistics``.
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


@deprecated_api(
    "This function is experimental and may be moved in a future release.",
    since="0.2.0",
)
def pattern_monotone_test(
    frame: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> Dict:
    """Test whether the missing data pattern is monotone (experimental).

    .. deprecated:: 0.2.0
        Experimental — may be moved or renamed.
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


@deprecated_api(
    "This function is experimental and may be moved in a future release.",
    since="0.2.0",
)
def missing_correlation_matrix(
    frame: pd.DataFrame,
    method: str = "pearson",
    missing_values: Optional[list] = None,
) -> pd.DataFrame:
    """Compute the pairwise nullity correlation matrix (experimental).

    .. deprecated:: 0.2.0
        Experimental — may be moved or renamed.
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
