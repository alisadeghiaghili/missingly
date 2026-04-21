import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from . import impute


def compare_imputations(df: pd.DataFrame, methods: list = None) -> pd.DataFrame:
    """Compare the performance of different imputation methods.

    Artificially masks 20% of values in each numeric column, applies each
    imputation method, and computes RMSE against the original values.
    Only numeric columns are used for RMSE evaluation; categorical columns
    are passed through to each imputer but excluded from scoring.

    Parameters
    ----------
    df : pd.DataFrame
        A complete dataframe (no missing values). Mixed numeric/categorical
        dtypes are supported.
    methods : list, optional
        A list of imputation functions to compare. Defaults to all seven
        built-in methods.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by method name with a single 'RMSE' column,
        sorted ascending (best method first).
    """
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
    if not numeric_cols:
        raise ValueError(
            "compare_imputations requires at least one numeric column for RMSE scoring."
        )

    # Mask 20% of each numeric column
    rng = np.random.default_rng(seed=42)
    df_missing = df.copy()
    for col in numeric_cols:
        n_mask = max(1, int(len(df_missing) * 0.2))
        mask_idx = rng.choice(df_missing.index, size=n_mask, replace=False)
        df_missing.loc[mask_idx, col] = np.nan

    results = {}
    for method in methods:
        df_imputed = method(df_missing)
        # Evaluate only on numeric columns
        rmse = np.sqrt(
            mean_squared_error(
                df[numeric_cols].values,
                df_imputed[numeric_cols].values,
            )
        )
        results[method.__name__] = rmse

    return (
        pd.DataFrame.from_dict(results, orient='index', columns=['RMSE'])
        .sort_values(by='RMSE')
    )
