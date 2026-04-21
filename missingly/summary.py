import pandas as pd
import numpy as np

def bind_shadow(df: pd.DataFrame, missing_values: list = None) -> pd.DataFrame:
    """Binds a shadow matrix to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to bind the shadow matrix to.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    pd.DataFrame
        A new dataframe with the shadow matrix bound to it.
    """
    if missing_values is None:
        shadow_df = df.isnull()
    else:
        shadow_df = df.isin(missing_values)
    shadow_df.columns = [f"{col}_NA" for col in df.columns]
    return pd.concat([df, shadow_df], axis=1)


def n_miss(df: pd.DataFrame, missing_values: list = None) -> int:
    """Counts the total number of missing values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to count missing values in.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    int
        The total number of missing values.
    """
    if missing_values is None:
        return df.isnull().sum().sum()
    else:
        return df.isin(missing_values).sum().sum()


def n_complete(df: pd.DataFrame, missing_values: list = None) -> int:
    """Counts the total number of non-missing values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to count non-missing values in.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    int
        The total number of non-missing values.
    """
    return df.size - n_miss(df, missing_values)


def pct_miss(df: pd.DataFrame, missing_values: list = None) -> float:
    """Calculates the percentage of missing values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the percentage of missing values in.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    float
        The percentage of missing values.
    """
    return (n_miss(df, missing_values) / df.size) * 100 if df.size > 0 else 0


def pct_complete(df: pd.DataFrame, missing_values: list = None) -> float:
    """Calculates the percentage of non-missing values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the percentage of non-missing values in.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    float
        The percentage of non-missing values.
    """
    return (n_complete(df, missing_values) / df.size) * 100 if df.size > 0 else 0


def miss_var_summary(df: pd.DataFrame, missing_values: list = None) -> pd.DataFrame:
    """Summarises missingness by variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarise.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    pd.DataFrame
        A dataframe with one row for each variable, and columns for the
        number and percentage of missing values.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = df.isin(missing_values).sum()
    miss_pct = (miss_counts / len(df)) * 100 if len(df) > 0 else 0
    summary_df = pd.DataFrame({
        'variable': df.columns,
        'n_miss': miss_counts,
        'pct_miss': miss_pct
    })
    return summary_df.reset_index(drop=True)


def miss_case_summary(df: pd.DataFrame, missing_values: list = None) -> pd.DataFrame:
    """Summarises missingness by case (row).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarise.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    pd.DataFrame
        A dataframe with one row for each case, and columns for the
        number and percentage of missing values.
    """
    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = df.isin(missing_values).sum(axis=1)
    miss_pct = (miss_counts / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    summary_df = pd.DataFrame({
        'case': df.index,
        'n_miss': miss_counts,
        'pct_miss': miss_pct
    })
    return summary_df.reset_index(drop=True)
