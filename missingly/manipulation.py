import pandas as pd
import numpy as np
from typing import Union, List, Dict, Callable

def replace_with_na(df: pd.DataFrame, replace: Dict[str, Union[List, object, Callable]]) -> pd.DataFrame:
    """Replaces specified values in a dataframe with NA.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify.
    replace : dict
        A dictionary where the keys are column names and the values
        are the values to replace with NA in that column. The values
        can be a single value, a list of values, or a function that
        returns a boolean.

    Returns
    -------
    pd.DataFrame
        A new dataframe with the specified values replaced with NA.
    """
    df_copy = df.copy()
    for col, condition in replace.items():
        if callable(condition):
            df_copy.loc[df_copy[col].apply(condition), col] = np.nan
        elif isinstance(condition, list):
            df_copy[col] = df_copy[col].replace(condition, np.nan)
        else:
            df_copy[col] = df_copy[col].replace([condition], np.nan)
    return df_copy


def replace_with_na_all(df: pd.DataFrame, condition: Callable) -> pd.DataFrame:
    """Replaces all values in a dataframe with NA if they meet a condition.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to modify.
    condition : callable
        A function that returns a boolean.

    Returns
    -------
    pd.DataFrame
        A new dataframe with the specified values replaced with NA.
    """
    return df.map(lambda x: np.nan if condition(x) else x)
