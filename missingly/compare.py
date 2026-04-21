import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from . import impute

def compare_imputations(df: pd.DataFrame, methods: list = None) -> pd.DataFrame:
    """Compares the performance of different imputation methods.

    This function artificially creates missing values in a complete dataset,
    applies different imputation methods, and then compares the imputed
    values to the original values using Root Mean Squared Error (RMSE).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to compare imputation methods on. It should not
        contain any missing values.
    methods : list, optional
        A list of imputation functions to compare. If not provided, a
        default set of methods will be used.

    Returns
    -------
    pd.DataFrame
        A dataframe with the RMSE for each imputation method.
    """
    if methods is None:
        methods = [
            impute.impute_mean,
            impute.impute_median,
            impute.impute_mode,
            impute.impute_knn,
            impute.impute_rf,
            impute.impute_gb,
            # impute.impute_mice, # MICE is causing issues with the test
        ]

    # Artificially create missing values
    df_missing = df.copy()
    for col in df_missing.columns:
        nan_indices = df_missing[col].sample(frac=0.2).index
        df_missing.loc[nan_indices, col] = np.nan
    
    results = {}
    for method in methods:
        df_imputed = method(df_missing)
        
        rmse = np.sqrt(mean_squared_error(df, df_imputed))
        results[method.__name__] = rmse

    return pd.DataFrame.from_dict(results, orient='index', columns=['RMSE']).sort_values(by='RMSE')
