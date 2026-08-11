"""Gower distance for mixed-type DataFrames.

Gower (1971) defines a similarity coefficient for observations that may
contain a mixture of numeric and categorical variables.  This module
exposes a single public helper, :func:`gower_distance`, that converts
that similarity into a distance matrix suitable for nearest-neighbour
imputation.

References
----------
Gower, J. C. (1971). A general coefficient of similarity and some of its
properties. *Biometrics*, 27(4), 857\u2013871.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["gower_distance"]

_LARGE_N_WARN = 10_000


def gower_distance(
    df: pd.DataFrame,
    cat_cols: Optional[list] = None,
) -> np.ndarray:
    """Compute the pairwise Gower distance matrix for a mixed-type DataFrame.

    Gower distance averages per-feature contributions:

    * **Numeric features**: normalised absolute difference
      ``|x_i - x_j| / range``, where ``range = max - min`` over all
      non-missing values in that column.  When ``range == 0`` the
      contribution is 0 (all values are identical).

    * **Categorical features**: 0 if the two values are equal, 1 otherwise.

    **Missing-value policy** \u2014 When one or both values are missing for a
    given feature the contribution of that feature is excluded from the
    average *and the denominator is decremented* so that the distance is
    still normalised to [0, 1].  Formally, each pairwise distance is::

        d(i, j) = sum(delta_k * s_k) / sum(delta_k)

    where ``delta_k = 1`` iff both ``x_ik`` and ``x_jk`` are observed,
    and ``s_k`` is the per-feature dissimilarity (see above).  When *no*
    feature is jointly observed the distance defaults to 1 (maximally
    dissimilar).

    This "skip-and-renormalise" strategy is preferred over treating any
    missing value as maximally dissimilar because it avoids artificially
    inflating the distances of rows that happen to have many NAs, which
    would bias nearest-neighbour selection during imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  May contain ``np.nan``, ``None``, or ``pd.NA`` in any
        column.
        The index is ignored; rows are treated positionally.
    cat_cols : list of str or None, default None
        Column names to treat as categorical.  If *None*, columns with
        ``object`` or ``category`` dtype are treated as categorical and
        all others as numeric.

    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        Symmetric distance matrix with values in [0, 1].  Diagonal is 0.

    Warnings
    --------
    For ``n_samples > 10 000`` this function computes an O(n\u00b2) matrix and
    may exhaust available memory.  Consider sub-sampling before calling.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({"num": [0.0, 1.0, 0.5], "cat": ["a", "b", "a"]})
    >>> D = gower_distance(df)
    >>> D.shape
    (3, 3)
    >>> float(D[0, 0])
    0.0
    >>> D[0, 1] > D[0, 2]   # row 0 is closer to row 2 than row 1
    True
    """
    n = len(df)
    if n > _LARGE_N_WARN:
        warnings.warn(
            f"gower_distance called with {n} rows. "
            "The O(n\u00b2) distance matrix may require significant memory. "
            "Consider sub-sampling to n \u2264 10 000 before calling.",
            UserWarning,
            stacklevel=2,
        )

    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Pre-compute column ranges for numeric features (ignoring NaN).
    ranges: dict[str, float] = {}
    for col in num_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        r = float(valid.max() - valid.min()) if len(valid) >= 2 else 0.0
        ranges[col] = r if r > 0 else 1.0  # avoid division by zero

    # Convert to numpy arrays for speed.
    num_arrays: dict[str, np.ndarray] = {
        col: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        for col in num_cols
    }
    cat_arrays: dict[str, np.ndarray] = {
        col: df[col].to_numpy(dtype=object) for col in cat_cols
    }

    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            total_s = 0.0
            valid_k = 0

            # Numeric contributions
            for col in num_cols:
                xi = num_arrays[col][i]
                xj = num_arrays[col][j]
                if np.isnan(xi) or np.isnan(xj):
                    continue
                total_s += abs(xi - xj) / ranges[col]
                valid_k += 1

            # Categorical contributions
            for col in cat_cols:
                xi = cat_arrays[col][i]
                xj = cat_arrays[col][j]
                # ``pd.isna`` handles pandas extension scalars such as ``pd.NA``.
                # Calling ``xi == xj`` before this check is unsafe because
                # ``pd.NA == value`` has no truth value.
                xi_miss = bool(pd.isna(xi))
                xj_miss = bool(pd.isna(xj))
                if xi_miss or xj_miss:
                    continue
                total_s += 0.0 if bool(xi == xj) else 1.0
                valid_k += 1

            d = (total_s / valid_k) if valid_k > 0 else 1.0
            dist[i, j] = d
            dist[j, i] = d

    return dist
