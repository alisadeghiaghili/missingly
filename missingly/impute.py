"""Imputation utilities for missing data.

This module provides two layers of imputation API:

1. **Stateless functions** (``impute_mean``, ``impute_median``, etc.)
   Accept a DataFrame, fit on that same DataFrame, and return a filled
   copy.  Convenient for exploration, but they re-fit on every call so
   they must **not** be used across train/test splits (data leakage).

2. **FittedImputer** (``make_imputer`` factory)
   A lightweight ``fit`` / ``transform`` / ``fit_transform`` wrapper
   around :class:`~missingly.transformer.MissinglyImputer` that uses
   the same strategy names as the stateless functions.  Use this
   whenever you need to fit on training data and transform test data
   separately — e.g. inside a scikit-learn ``Pipeline``.

Key design decisions
--------------------
* Python ``None`` in object-dtype columns is normalised to ``np.nan``
  before any sklearn estimator sees the data.
* Categorical columns are handled separately for ML-based imputers:
  - Numeric columns → imputed with the provided regressor.
  - Categorical columns → imputed with a classifier, avoiding the
    error of treating category codes as continuous values.
* ``GradientBoostingRegressor`` / ``GradientBoostingClassifier`` do not
  accept NaN in feature matrices.  Any remaining NaN in the feature
  side is filled with column means computed from the training rows.

Large-data warnings
-------------------
Functions that are O(n²) or slow on large DataFrames emit a
``UserWarning`` when the input has more than ``_LARGE_DF_ROW_THRESHOLD``
rows.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

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


_LARGE_DF_ROW_THRESHOLD = 50_000
# Threshold: warn when n_cat > n_num AND n_neighbors > this value.
_KNN_CAT_NEIGHBORS_THRESHOLD = 5


def _warn_if_large(df: pd.DataFrame, method_name: str) -> None:
    if len(df) > _LARGE_DF_ROW_THRESHOLD:
        warnings.warn(
            f"{method_name}: DataFrame has {len(df):,} rows which may result in "
            f"very long runtimes or high memory usage. "
            f"Consider sampling your data or using impute_mean / impute_median "
            f"for large datasets.",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_knn_heavy_categorical(df: pd.DataFrame, n_neighbors: int) -> None:
    n_cat = len(df.select_dtypes(exclude=[np.number]).columns)
    n_num = len(df.select_dtypes(include=[np.number]).columns)
    if n_cat > n_num and n_neighbors > _KNN_CAT_NEIGHBORS_THRESHOLD:
        warnings.warn(
            f"KNN imputation may be unreliable: the DataFrame has {n_cat} categorical "
            f"column(s) but only {n_num} numeric column(s), and n_neighbors={n_neighbors} "
            f"(> {_KNN_CAT_NEIGHBORS_THRESHOLD}). "
            "Euclidean distance over ordinal-encoded categoricals is not statistically "
            "meaningful and degrades as n_neighbors grows. "
            "Consider impute_rf(), impute_gb(), or impute_mice() for heavy-categorical datasets.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    obj_cols = result.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        result[obj_cols] = result[obj_cols].where(
            result[obj_cols].notna(), other=np.nan
        )
    return result


def _split_encode(df: pd.DataFrame):
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

    df_work = df_work.astype(float)
    return df_work, cat_cols, num_cols, encoder, cat_dtypes


def _decode(
    df_imputed: pd.DataFrame,
    cat_cols: list,
    encoder,
    cat_dtypes: dict,
) -> pd.DataFrame:
    if not cat_cols or encoder is None:
        return df_imputed

    result = df_imputed.copy()
    rounded = np.round(result[cat_cols].to_numpy()).clip(0).copy()
    decoded = encoder.inverse_transform(rounded)
    for i, col in enumerate(cat_cols):
        result[col] = decoded[:, i]
        try:
            result[col] = result[col].astype(cat_dtypes[col])
        except (ValueError, TypeError):
            pass
    return result


def _fill_nan_with_col_means(X: np.ndarray) -> np.ndarray:
    X_out = X.copy()
    col_means = np.nanmean(X_out, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X_out)
    X_out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X_out


def _impute_column_by_column(
    df: pd.DataFrame,
    regressor,
    classifier,
    random_state: int = 0,
    max_iter: int = 1,
    fill_feature_nan: bool = False,
) -> pd.DataFrame:
    cat_cols = set(df.select_dtypes(exclude=[np.number]).columns)
    result = df.copy()

    for _ in range(max_iter):
        for col in df.columns:
            missing_mask = result[col].isnull()
            if not missing_mask.any():
                continue

            df_enc, _, _, enc_tmp, _ = _split_encode(result)
            feature_cols = [c for c in df.columns if c != col]
            if not feature_cols:
                continue

            X = df_enc[feature_cols].values
            y_series = result[col]
            train_mask = ~missing_mask
            if train_mask.sum() < 2:
                continue

            X_train = X[train_mask]
            X_pred = X[missing_mask]

            if fill_feature_nan:
                X_train = _fill_nan_with_col_means(X_train)
                X_pred = _fill_nan_with_col_means(X_pred)

            if col in cat_cols:
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


# ---------------------------------------------------------------------------
# Stateless imputation functions
# ---------------------------------------------------------------------------

def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using the column mean (numeric) / mode (categorical)."""
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
    """Impute missing values using the column median (numeric) / mode (categorical)."""
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
    """Impute missing values using the most frequent value of each column."""
    df = _normalize_missing(df)
    imputer = SimpleImputer(strategy="most_frequent")
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbors (KNN)."""
    _warn_if_large(df, "impute_knn")
    _warn_if_knn_heavy_categorical(df, n_neighbors)
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
    n_imputations: int = 1,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Impute missing values using MICE (IterativeImputer).

    When ``n_imputations > 1`` runs independent chains each with a distinct
    random seed so the returned DataFrames differ from one another.
    """
    if n_imputations < 1:
        raise ValueError(f"n_imputations must be >= 1; got {n_imputations}.")

    _warn_if_large(df, "impute_mice")
    df_norm = _normalize_missing(df)

    def _single_chain(seed: int) -> pd.DataFrame:
        df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df_norm)
        # Use sample_posterior=True so each chain draws different samples
        # from the posterior and produces distinct imputed values.
        if estimator is None:
            est = BayesianRidge()
            use_sample_posterior = True
        else:
            est = estimator
            use_sample_posterior = False
        imputer = IterativeImputer(
            estimator=est,
            max_iter=max_iter,
            random_state=seed,
            imputation_order="roman",
            sample_posterior=use_sample_posterior,
        )
        imputed_array = imputer.fit_transform(df_work)
        df_imputed = pd.DataFrame(
            imputed_array, index=df_norm.index, columns=df_work.columns
        )
        return _decode(df_imputed, cat_cols, encoder, cat_dtypes)

    if n_imputations == 1:
        return _single_chain(random_state)

    return [_single_chain(random_state + i) for i in range(n_imputations)]


def impute_rf(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    **rf_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Random Forest."""
    _warn_if_large(df, "impute_rf")
    df = _normalize_missing(df)
    regressor = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    classifier = RandomForestClassifier(random_state=random_state, **rf_kwargs)
    return _impute_column_by_column(
        df, regressor=regressor, classifier=classifier,
        random_state=random_state, max_iter=max_iter,
        fill_feature_nan=False,
    )


def impute_gb(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    **gb_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Gradient Boosting."""
    _warn_if_large(df, "impute_gb")
    df = _normalize_missing(df)
    regressor = GradientBoostingRegressor(random_state=random_state, **gb_kwargs)
    classifier = GradientBoostingClassifier(random_state=random_state, **gb_kwargs)
    return _impute_column_by_column(
        df, regressor=regressor, classifier=classifier,
        random_state=random_state, max_iter=max_iter,
        fill_feature_nan=True,
    )


# ---------------------------------------------------------------------------
# FittedImputer — fit / transform interface
# ---------------------------------------------------------------------------

class FittedImputer:
    """Lightweight fit/transform wrapper for missingly imputation strategies."""

    def __init__(self, strategy: str = "mean", **kwargs) -> None:
        from .transformer import MissinglyImputer
        self.strategy = strategy
        self._imputer = MissinglyImputer(strategy=strategy, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self._imputer._is_fitted

    def fit(self, X: pd.DataFrame, y=None) -> "FittedImputer":
        self._imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "FittedImputer.transform() called before fit(). "
                "Call fit(X_train) first."
            )
        return self._imputer.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "unfitted"
        return f"FittedImputer(strategy={self.strategy!r}, {fitted})"


def make_imputer(strategy: str = "mean", **kwargs) -> FittedImputer:
    """Factory function that returns an unfitted :class:`FittedImputer`."""
    return FittedImputer(strategy=strategy, **kwargs)
