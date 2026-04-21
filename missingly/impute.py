import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


def _split_encode(df: pd.DataFrame):
    """Split df into numeric and categorical parts, ordinal-encode categoricals.

    Returns
    -------
    df_work : pd.DataFrame
        All columns as float64 — numerics unchanged, categoricals encoded.
    cat_cols : list[str]
        Names of categorical columns.
    num_cols : list[str]
        Names of numeric columns.
    encoder : OrdinalEncoder or None
        Fitted encoder (None if no categorical columns).
    cat_dtypes : dict[str, dtype]
        Original dtypes of categorical columns (for decode).
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_work = df.copy()
    encoder = None
    cat_dtypes = {}

    if cat_cols:
        cat_dtypes = {c: df[c].dtype for c in cat_cols}
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        df_work[cat_cols] = encoder.fit_transform(df[cat_cols])

    # ensure all float so sklearn imputers don't complain
    df_work = df_work.astype(float)
    return df_work, cat_cols, num_cols, encoder, cat_dtypes


def _decode(df_imputed: pd.DataFrame, cat_cols: list, encoder, cat_dtypes: dict) -> pd.DataFrame:
    """Inverse-transform ordinal-encoded columns back to original categories."""
    if not cat_cols or encoder is None:
        return df_imputed

    result = df_imputed.copy()
    # Round encoded values to nearest integer before inverse transform
    result[cat_cols] = np.round(result[cat_cols]).clip(0)
    decoded = encoder.inverse_transform(result[cat_cols])
    for i, col in enumerate(cat_cols):
        result[col] = decoded[:, i]
        # restore original dtype where possible
        try:
            result[col] = result[col].astype(cat_dtypes[col])
        except (ValueError, TypeError):
            pass
    return result


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the mean of each numeric column.

    Non-numeric columns are imputed with their most frequent value
    (same behaviour as impute_mode for categoricals).

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
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy='mean')
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        result[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return result


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the median of each numeric column.

    Non-numeric columns are imputed with their most frequent value.

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
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        result[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
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

    Categorical columns are automatically ordinal-encoded before imputation
    and decoded back to their original categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
    n_neighbors : int, optional
        The number of neighbors to use for imputation. Default is 5.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_work.columns)
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)


def impute_mice(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    estimator=None,
) -> pd.DataFrame:
    """Impute missing values using Multiple Imputation by Chained Equations (MICE).

    Uses sklearn's IterativeImputer with BayesianRidge as the default
    estimator. Categorical columns are automatically ordinal-encoded before
    imputation and decoded back to their original categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
    max_iter : int, optional
        Maximum number of imputation rounds. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    estimator : sklearn estimator, optional
        The estimator to use for each round-robin imputation step.
        Defaults to BayesianRidge().

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)

    if estimator is None:
        estimator = BayesianRidge()

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        imputation_order='roman',
    )
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_work.columns)
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)


def impute_rf(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    **rf_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Random Forest via IterativeImputer.

    Categorical columns are automatically ordinal-encoded before imputation
    and decoded back to their original categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
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
    """
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)
    estimator = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
    )
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_work.columns)
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)


def impute_gb(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    **gb_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Gradient Boosting via IterativeImputer.

    Categorical columns are automatically ordinal-encoded before imputation
    and decoded back to their original categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.
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
    """
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)
    estimator = GradientBoostingRegressor(random_state=random_state, **gb_kwargs)
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
    )
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_work.columns)
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)
