"""Imputation comparison utilities.

Provides :func:`compare_imputations` which benchmarks multiple imputation
strategies on a complete DataFrame by artificially masking a fraction of
values and measuring reconstruction accuracy.

Scoring
-------
* **Numeric columns** → RMSE (root mean squared error) on the masked cells.
* **Categorical columns** → accuracy (fraction of correctly imputed values)
  on the masked cells.

The final ranking uses a weighted combination:
  - If both numeric and categorical columns exist, the result table
    contains three columns: ``RMSE``, ``Accuracy``, and ``Score``.
    ``Score`` is a normalised composite (lower RMSE + higher accuracy =
    lower score), enabling a single-column sort.
  - If only numeric columns exist, the result contains only ``RMSE``.
  - If only categorical columns exist, the result contains only
    ``Accuracy`` (inverted for sort: lower is better).

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

from . import impute


def compare_imputations(
    df: pd.DataFrame,
    methods: Optional[List] = None,
    mask_frac: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare the performance of different imputation methods.

    Artificially masks a fraction of values in each column, applies each
    imputation method, and evaluates reconstruction accuracy.

    * Numeric columns are scored with **RMSE** (lower is better).
    * Categorical columns are scored with **accuracy** (higher is better).

    When both column types are present, a composite ``Score`` column is
    added that normalises and combines both metrics (lower = better) so
    results can be sorted into a single ranking.

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** dataframe (no missing values) with at least one
        column.  Mixed numeric/categorical dtypes are supported.
    methods : list of callables, optional
        Imputation functions to compare.  Each must accept a DataFrame and
        return a fully-imputed DataFrame.  Defaults to all seven built-in
        methods: mean, median, mode, knn, mice, rf, gb.
    mask_frac : float, optional
        Fraction of values to mask per column (default 0.20, i.e. 20%).
        Must be in (0, 1).
    random_state : int, optional
        Random seed for reproducible masking.  Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by method name, sorted ascending by ``Score``
        (or ``RMSE`` / ``1 - Accuracy`` when only one column type exists).
        Columns present depend on the input data:

        * ``RMSE`` — present when numeric columns exist.
        * ``Accuracy`` — present when categorical columns exist.
        * ``Score`` — present when both column types exist (composite metric).

    Raises
    ------
    ValueError
        If *df* has missing values (a complete DataFrame is required).
    ValueError
        If *mask_frac* is not in (0, 1).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'age': [25, 30, 35, 40], 'city': ['A','B','A','B']})
    >>> compare_imputations(df)
    """
    if df.isnull().any().any():
        raise ValueError(
            "compare_imputations requires a complete DataFrame (no missing values). "
            "Call df.dropna() first."
        )
    if not (0.0 < mask_frac < 1.0):
        raise ValueError(f"mask_frac must be in (0, 1); got {mask_frac!r}")

    if methods is None:
        methods = [
            impute.impute_mean,
            impute.impute_median,
            impute.impute_mode,
            impute.impute_knn,
            impute.impute_mice,
            impute.impute_rf,
            impute.impute_gb,
        ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols and not cat_cols:
        raise ValueError("DataFrame has no columns to evaluate.")

    rng = np.random.default_rng(seed=random_state)

    # Build per-column masks
    masks: dict = {}
    for col in numeric_cols + cat_cols:
        n_mask = max(1, int(len(df) * mask_frac))
        idx = rng.choice(df.index, size=n_mask, replace=False)
        masks[col] = idx

    # Apply masks to create the missing DataFrame
    df_missing = df.copy()
    for col, idx in masks.items():
        df_missing.loc[idx, col] = np.nan

    results = {}
    for method in methods:
        df_imputed = method(df_missing)
        row: dict = {}

        if numeric_cols:
            rmse_vals = []
            for col in numeric_cols:
                idx = masks[col]
                true_vals = df.loc[idx, col].values.astype(float)
                pred_vals = df_imputed.loc[idx, col].values.astype(float)
                rmse_vals.append(
                    np.sqrt(mean_squared_error(true_vals, pred_vals))
                )
            row["RMSE"] = float(np.mean(rmse_vals))

        if cat_cols:
            acc_vals = []
            for col in cat_cols:
                idx = masks[col]
                true_vals = df.loc[idx, col].values
                pred_vals = df_imputed.loc[idx, col].values
                acc_vals.append(accuracy_score(true_vals, pred_vals))
            row["Accuracy"] = float(np.mean(acc_vals))

        results[method.__name__] = row

    result_df = pd.DataFrame.from_dict(results, orient="index")

    # Build composite score for mixed DataFrames
    if numeric_cols and cat_cols:
        # Normalise RMSE to [0,1] range, invert accuracy so lower = better
        rmse_min, rmse_max = result_df["RMSE"].min(), result_df["RMSE"].max()
        if rmse_max > rmse_min:
            norm_rmse = (result_df["RMSE"] - rmse_min) / (rmse_max - rmse_min)
        else:
            norm_rmse = pd.Series(0.0, index=result_df.index)
        result_df["Score"] = (norm_rmse + (1.0 - result_df["Accuracy"])) / 2.0
        return result_df.sort_values(by="Score")
    elif numeric_cols:
        return result_df.sort_values(by="RMSE")
    else:
        # Only categorical — sort by accuracy descending
        return result_df.sort_values(by="Accuracy", ascending=False)
