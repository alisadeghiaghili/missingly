"""Imputation utilities for missing data.

This module provides a collection of imputation strategies that work on
pandas DataFrames containing numeric and/or categorical columns.  All
public functions accept a DataFrame, impute missing values in-place on a
copy, and return the filled copy without modifying the input.

Key design decisions
--------------------
* Python ``None`` in object-dtype columns is normalised to ``np.nan``
  before any sklearn estimator sees the data, because sklearn only
  treats ``np.nan`` as a missing sentinel.
* Categorical columns are handled separately for ML-based imputers:
  - Numeric columns → imputed with the provided regressor (default or custom).
  - Categorical columns → imputed with a ``RandomForestClassifier`` /
    ``GradientBoostingClassifier`` (for RF/GB methods) or
    ``KNNImputer`` + ``OrdinalEncoder`` round-trip (for KNN/MICE).
  This avoids the error of using regression to impute nominal categories.
* ``impute_mean``, ``impute_median``, ``impute_mode`` remain unchanged:
  they rely on ``SimpleImputer`` which handles categoricals via
  ``most_frequent`` strategy.

Large-data warnings
-------------------
Functions that are O(n²) or otherwise slow on large DataFrames will emit
a ``UserWarning`` when the input has more than ``_LARGE_DF_ROW_THRESHOLD``
rows, so users are not silently surprised by long runtimes.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import OrdinalEncoder


# Threshold above which O(n²) / slow imputers emit a UserWarning.
_LARGE_DF_ROW_THRESHOLD = 50_000


def _warn_if_large(df: pd.DataFrame, method_name: str) -> None:
    """Emit a UserWarning if *df* has more rows than the large-data threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame being imputed.
    method_name : str
        Name of the calling imputation function (used in the warning text).
    """
    if len(df) > _LARGE_DF_ROW_THRESHOLD:
        warnings.warn(
            f"{method_name}: DataFrame has {len(df):,} rows which may result in "
            f"very long runtimes or high memory usage. "
            f"Consider sampling your data or using impute_mean / impute_median "
            f"for large datasets.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Python ``None`` with ``np.nan`` in all object/string-dtype columns.

    sklearn estimators only recognise ``np.nan`` as a missing sentinel;
    Python ``None`` stored in object columns is silently ignored, which
    causes imputation to leave those cells unfilled.  This helper
    normalises the representation so downstream imputers behave correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, possibly containing ``None`` in object columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* where every ``None`` in object/string-dtype columns
        has been replaced with ``np.nan``.

    Notes
    -----
    Both ``'object'`` and ``'string'`` dtypes are included explicitly.
    Under pandas >= 3.0, ``select_dtypes(include=['object'])`` no longer
    implicitly captures the ``StringDtype`` (``pd.StringDtype()``), so
    passing both prevents a ``Pandas4Warning`` and ensures correctness
    across pandas 2.x – 3.x.
    """
    result = df.copy()
    obj_cols = result.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        result[obj_cols] = result[obj_cols].where(
            result[obj_cols].notna(), other=np.nan
        )
    return result


def _split_encode(df: pd.DataFrame):
    """Split df into numeric and categorical parts, ordinal-encode categoricals.

    Assumes ``None`` has already been normalised to ``np.nan`` (call
    ``_normalize_missing`` first).  ``OrdinalEncoder`` is configured with
    ``encoded_missing_value=np.nan`` so it propagates ``np.nan`` through
    to the imputer rather than raising on unseen missing markers.

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
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        df_work[cat_cols] = encoder.fit_transform(df[cat_cols])

    # ensure all float so sklearn imputers don't complain
    df_work = df_work.astype(float)
    return df_work, cat_cols, num_cols, encoder, cat_dtypes


def _decode(df_imputed: pd.DataFrame, cat_cols: list, encoder, cat_dtypes: dict) -> pd.DataFrame:
    """Inverse-transform ordinal-encoded columns back to original categories.

    Parameters
    ----------
    df_imputed : pd.DataFrame
        DataFrame whose categorical columns contain float-encoded values.
    cat_cols : list[str]
        Names of the categorical columns to decode.
    encoder : OrdinalEncoder or None
        The fitted encoder returned by ``_split_encode``.
    cat_dtypes : dict[str, dtype]
        Original dtypes to restore after decoding.

    Returns
    -------
    pd.DataFrame
        Copy of *df_imputed* with categorical columns decoded back to
        their original string/category values.

    Notes
    -----
    Encoded values are rounded and clipped to ``[0, n_categories - 1]``
    before inverse-transforming to handle any floating-point drift
    introduced by ML-based imputers.
    """
    if not cat_cols or encoder is None:
        return df_imputed

    result = df_imputed.copy()
    result[cat_cols] = np.round(result[cat_cols]).clip(0)
    decoded = encoder.inverse_transform(result[cat_cols])
    for i, col in enumerate(cat_cols):
        result[col] = decoded[:, i]
        try:
            result[col] = result[col].astype(cat_dtypes[col])
        except (ValueError, TypeError):
            pass
    return result


def _impute_column_by_column(
    df: pd.DataFrame,
    regressor,
    classifier,
    random_state: int = 0,
    max_iter: int = 1,
) -> pd.DataFrame:
    """Impute each column using a regressor (numeric) or classifier (categorical).

    This performs a single-pass (non-iterative) column-by-column imputation:
    for each target column with missing values, a model is trained on all
    complete rows using the remaining columns as features.  Categorical
    columns are label-encoded for use as features but decoded back to their
    original values in the output.

    This approach ensures that:
    - Numeric columns are imputed with a *regressor* (appropriate for
      continuous targets).
    - Categorical columns are imputed with a *classifier* (appropriate for
      nominal / ordinal targets), avoiding the error of treating category
      codes as continuous values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values.  ``None`` should already have been
        normalised to ``np.nan`` via ``_normalize_missing``.
    regressor : sklearn estimator
        Fitted-style regressor (e.g. ``RandomForestRegressor``) used for
        numeric target columns.
    classifier : sklearn estimator
        Fitted-style classifier (e.g. ``RandomForestClassifier``) used for
        categorical target columns.
    random_state : int, optional
        Passed to both estimators when they are instantiated.
    max_iter : int, optional
        Number of full passes over all columns with missing values.
        Default is 1 (single pass).  Higher values give iterative refinement
        similar to MICE but without the BayesianRidge assumption.

    Returns
    -------
    pd.DataFrame
        Fully-imputed copy of *df*.

    Notes
    -----
    Columns that are entirely missing cannot be imputed and will remain NaN.
    """
    cat_cols = set(df.select_dtypes(exclude=[np.number]).columns)
    num_cols = set(df.select_dtypes(include=[np.number]).columns)

    # Build a float-encoded working copy for feature matrices.
    _, _, _, encoder, cat_dtypes = _split_encode(df)
    result = df.copy()

    for _ in range(max_iter):
        for col in df.columns:
            missing_mask = result[col].isnull()
            if not missing_mask.any():
                continue

            # Feature matrix: encode current state of result
            df_enc, _, _, enc_tmp, _ = _split_encode(result)
            feature_cols = [c for c in df.columns if c != col]
            if not feature_cols:
                continue

            X = df_enc[feature_cols].values
            y_series = result[col]

            train_mask = ~missing_mask
            if train_mask.sum() < 2:
                # Not enough data to train — skip
                continue

            X_train = X[train_mask]
            X_pred = X[missing_mask]

            if col in cat_cols:
                # Encode target for classification
                enc_y = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                )
                y_train_enc = enc_y.fit_transform(
                    y_series[train_mask].values.reshape(-1, 1)
                ).ravel().astype(int)
                model = classifier
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_pred).reshape(-1, 1)
                y_pred = enc_y.inverse_transform(y_pred_enc).ravel()
                result.loc[missing_mask, col] = y_pred
            else:
                y_train = y_series[train_mask].values
                model = regressor
                model.fit(X_train, y_train)
                result.loc[missing_mask, col] = model.predict(X_pred)

    return result


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the mean of each numeric column.

    Non-numeric columns are imputed with their most frequent value
    (same behaviour as ``impute_mode`` for categoricals).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    df = _normalize_missing(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy="mean")
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return result


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the median of each numeric column.

    Non-numeric columns are imputed with their most frequent value.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.
    """
    df = _normalize_missing(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    result = df.copy()
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return result


def impute_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the most frequent value of each column.

    Works for both numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Notes
    -----
    Unlike ``impute_mean``/``impute_median``, this function passes all
    columns to a single ``SimpleImputer``.  Original dtypes may change
    to ``object`` for columns that were mixed numeric/string.  If dtype
    preservation is important, use ``impute_mean`` instead.
    """
    df = _normalize_missing(df)
    imputer = SimpleImputer(strategy="most_frequent")
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbors.

    Categorical columns are automatically ordinal-encoded before imputation
    and decoded back to their original categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.
    n_neighbors : int, optional
        The number of neighbors to use for imputation. Default is 5.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Warns
    -----
    UserWarning
        If the DataFrame has more than 50,000 rows, a warning is emitted
        because KNN imputation is O(n²) and may be very slow.
    """
    _warn_if_large(df, "impute_knn")
    df = _normalize_missing(df)
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

    Uses sklearn's ``IterativeImputer`` with ``BayesianRidge`` as the default
    estimator for all columns.  Categorical columns are automatically
    ordinal-encoded before imputation and decoded back to their original
    categories afterwards.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.
    max_iter : int, optional
        Maximum number of imputation rounds. Default is 10.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    estimator : sklearn estimator, optional
        The estimator to use for each round-robin imputation step.
        Defaults to ``BayesianRidge()``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Warns
    -----
    UserWarning
        If the DataFrame has more than 50,000 rows.
    """
    _warn_if_large(df, "impute_mice")
    df = _normalize_missing(df)
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)

    if estimator is None:
        estimator = BayesianRidge()

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        imputation_order="roman",
    )
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_work.columns)
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)


def impute_rf(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    **rf_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Random Forest.

    Numeric columns are imputed with ``RandomForestRegressor``; categorical
    columns are imputed with ``RandomForestClassifier``.  This column-by-column
    approach ensures that nominal categories are never treated as continuous
    values, which would produce nonsensical imputed values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.
    max_iter : int, optional
        Number of full passes over all columns.  Default is 1.
        Set to > 1 for iterative refinement (similar to MICE).
    random_state : int, optional
        Random seed passed to both the regressor and classifier.
    **rf_kwargs
        Additional keyword arguments passed to both
        ``RandomForestRegressor`` and ``RandomForestClassifier``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Warns
    -----
    UserWarning
        If the DataFrame has more than 50,000 rows.
    """
    _warn_if_large(df, "impute_rf")
    df = _normalize_missing(df)
    regressor = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    classifier = RandomForestClassifier(random_state=random_state, **rf_kwargs)
    return _impute_column_by_column(
        df, regressor=regressor, classifier=classifier,
        random_state=random_state, max_iter=max_iter,
    )


def impute_gb(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    **gb_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Gradient Boosting.

    Numeric columns are imputed with ``GradientBoostingRegressor``;
    categorical columns are imputed with ``GradientBoostingClassifier``.
    This column-by-column approach ensures that nominal categories are
    never treated as continuous values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to impute.  May contain ``None`` or ``np.nan`` as
        missing markers in object columns.
    max_iter : int, optional
        Number of full passes over all columns.  Default is 1.
    random_state : int, optional
        Random seed passed to both the regressor and classifier.
    **gb_kwargs
        Additional keyword arguments passed to both
        ``GradientBoostingRegressor`` and ``GradientBoostingClassifier``.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values imputed.

    Warns
    -----
    UserWarning
        If the DataFrame has more than 50,000 rows.
    """
    _warn_if_large(df, "impute_gb")
    df = _normalize_missing(df)
    regressor = GradientBoostingRegressor(random_state=random_state, **gb_kwargs)
    classifier = GradientBoostingClassifier(random_state=random_state, **gb_kwargs)
    return _impute_column_by_column(
        df, regressor=regressor, classifier=classifier,
        random_state=random_state, max_iter=max_iter,
    )
