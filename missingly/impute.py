import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the mean of each column.

    Only applicable to numeric columns. Non-numeric columns are returned
    unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy='mean')
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return result


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the median of each column.

    Only applicable to numeric columns. Non-numeric columns are returned
    unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return result


def impute_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the most frequent value of each column.

    Works for both numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbors.

    Only applicable to numeric columns. Raises ValueError if any
    non-numeric columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute. Must contain only numeric columns.
    n_neighbors : int, optional
        The number of neighbors to use for imputation. Default is 5.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Raises
    ------
    ValueError
        If the dataframe contains non-numeric columns.
    """
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"KNN imputation requires numeric columns only. "
            f"Non-numeric columns found: {non_numeric}. "
            f"Please encode or drop them before calling impute_knn()."
        )
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_mice(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    estimator=None,
) -> pd.DataFrame:
    """Impute missing values using Multiple Imputation by Chained Equations (MICE).

    Uses sklearn's IterativeImputer with BayesianRidge as the default
    estimator, which is the standard MICE-equivalent in the Python
    ecosystem. Each feature is imputed as a function of all other features
    in a round-robin fashion for ``max_iter`` iterations.

    Only applicable to numeric columns. Raises ValueError if any
    non-numeric columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute. Must contain only numeric columns.
    max_iter : int, optional
        Maximum number of imputation rounds. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    estimator : sklearn estimator, optional
        The estimator to use for each round-robin imputation step.
        Defaults to BayesianRidge(), which is the standard MICE estimator.
        You may pass any sklearn-compatible regressor, e.g.
        RandomForestRegressor().

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Raises
    ------
    ValueError
        If the dataframe contains non-numeric columns.
    """
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"MICE imputation requires numeric columns only. "
            f"Non-numeric columns found: {non_numeric}. "
            f"Please encode or drop them before calling impute_mice()."
        )

    if estimator is None:
        estimator = BayesianRidge()

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        imputation_order='roman',  # left-to-right, standard MICE order
    )
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_rf(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    **rf_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Random Forest via IterativeImputer.

    Only applicable to numeric columns. Raises ValueError if any
    non-numeric columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute. Must contain only numeric columns.
    max_iter : int, optional
        Maximum number of imputation rounds. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    **rf_kwargs
        Additional keyword arguments passed to RandomForestRegressor.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Raises
    ------
    ValueError
        If the dataframe contains non-numeric columns.
    """
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Random Forest imputation requires numeric columns only. "
            f"Non-numeric columns found: {non_numeric}. "
            f"Please encode or drop them before calling impute_rf()."
        )
    estimator = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
    )
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_gb(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    **gb_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Gradient Boosting via IterativeImputer.

    Only applicable to numeric columns. Raises ValueError if any
    non-numeric columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute. Must contain only numeric columns.
    max_iter : int, optional
        Maximum number of imputation rounds. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    **gb_kwargs
        Additional keyword arguments passed to GradientBoostingRegressor.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Raises
    ------
    ValueError
        If the dataframe contains non-numeric columns.
    """
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Gradient Boosting imputation requires numeric columns only. "
            f"Non-numeric columns found: {non_numeric}. "
            f"Please encode or drop them before calling impute_gb()."
        )
    estimator = GradientBoostingRegressor(random_state=random_state, **gb_kwargs)
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
    )
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)
