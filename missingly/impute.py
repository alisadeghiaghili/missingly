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


_LARGE_DF_ROW_THRESHOLD = 50_000


def _warn_if_large(df: pd.DataFrame, method_name: str) -> None:
    """Emit a UserWarning when *df* exceeds the large-data row threshold.

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
    """
    result = df.copy()
    obj_cols = result.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        result[obj_cols] = result[obj_cols].where(
            result[obj_cols].notna(), other=np.nan
        )
    return result


def _split_encode(df: pd.DataFrame):
    """Split df into numeric and categorical parts; ordinal-encode categoricals.

    Returns
    -------
    df_work : pd.DataFrame
        All columns as float64.
    cat_cols : list[str]
    num_cols : list[str]
    encoder : OrdinalEncoder or None
    cat_dtypes : dict[str, dtype]
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

    df_work = df_work.astype(float)
    return df_work, cat_cols, num_cols, encoder, cat_dtypes


def _decode(
    df_imputed: pd.DataFrame,
    cat_cols: list,
    encoder,
    cat_dtypes: dict,
) -> pd.DataFrame:
    """Inverse-transform ordinal-encoded columns back to original categories.

    Parameters
    ----------
    df_imputed : pd.DataFrame
    cat_cols : list[str]
    encoder : OrdinalEncoder or None
    cat_dtypes : dict[str, dtype]

    Returns
    -------
    pd.DataFrame
    """
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
    """Fill NaN values in a 2-D float array with column means.

    Used internally to make feature matrices safe for
    ``GradientBoostingRegressor`` / ``GradientBoostingClassifier``.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray
    """
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
    """Impute each column using a regressor (numeric) or classifier (categorical).

    Parameters
    ----------
    df : pd.DataFrame
    regressor : sklearn estimator
    classifier : sklearn estimator
    random_state : int
    max_iter : int
    fill_feature_nan : bool
        Fill NaN in feature matrix with column means before fit/predict.

    Returns
    -------
    pd.DataFrame
    """
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
    """Impute missing values using the column mean (numeric) / mode (categorical).

    .. warning::
        This function fits and transforms the *same* DataFrame.  Do **not**
        use it across train/test splits.  Use :func:`make_imputer` instead
        to obtain a ``FittedImputer`` with a proper ``fit`` / ``transform``
        interface.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
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
    """Impute missing values using the column median (numeric) / mode (categorical).

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
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

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = _normalize_missing(df)
    imputer = SimpleImputer(strategy="most_frequent")
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, index=df.index, columns=df.columns)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbors.

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame
    n_neighbors : int, optional
        Default 5.

    Returns
    -------
    pd.DataFrame
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
    """Impute missing values using Multiple Imputation by Chained Equations.

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame
    max_iter : int, optional
        Default 10.
    random_state : int, optional
        Default 0.
    estimator : sklearn estimator, optional
        Defaults to ``BayesianRidge()``.

    Returns
    -------
    pd.DataFrame
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
    """Impute missing values using Random Forest (regressor + classifier).

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame
    max_iter : int, optional
        Default 1.
    random_state : int, optional
    **rf_kwargs
        Forwarded to both ``RandomForestRegressor`` and
        ``RandomForestClassifier``.

    Returns
    -------
    pd.DataFrame
    """
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
    """Impute missing values using Gradient Boosting (regressor + classifier).

    .. warning::
        Fits and transforms the same DataFrame.  Use :func:`make_imputer`
        for proper train/test separation.

    Parameters
    ----------
    df : pd.DataFrame
    max_iter : int, optional
        Default 1.
    random_state : int, optional
    **gb_kwargs
        Forwarded to both ``GradientBoostingRegressor`` and
        ``GradientBoostingClassifier``.

    Returns
    -------
    pd.DataFrame
    """
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
    """Lightweight fit/transform wrapper for missingly imputation strategies.

    Unlike the stateless ``impute_*`` functions, ``FittedImputer`` separates
    the fitting step (learning statistics from training data) from the
    transform step (applying those statistics to new data).  This prevents
    data leakage when imputing train and test sets.

    Under the hood this delegates to
    :class:`~missingly.transformer.MissinglyImputer`, which is a full
    scikit-learn ``BaseEstimator`` / ``TransformerMixin`` and can be
    embedded directly in a ``sklearn.pipeline.Pipeline``.

    Parameters
    ----------
    strategy : str
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"knn"``,
        ``"mice"``, ``"rf"``, ``"gb"``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~missingly.transformer.MissinglyImputer`
        (e.g. ``n_neighbors=3`` for KNN, ``random_state=42`` for RF).

    Attributes
    ----------
    strategy : str
    is_fitted : bool
        ``True`` after :meth:`fit` has been called.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import make_imputer
    >>>
    >>> X_train = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
    >>> X_test  = pd.DataFrame({'a': [np.nan, 2.0],      'b': [6.0, np.nan]})
    >>>
    >>> imp = make_imputer('mean')
    >>> imp.fit(X_train)
    >>> X_train_clean = imp.transform(X_train)   # uses train means
    >>> X_test_clean  = imp.transform(X_test)    # still uses train means — no leakage

    Notes
    -----
    For full sklearn pipeline integration use
    :class:`~missingly.transformer.MissinglyImputer` directly, which
    inherits from ``BaseEstimator`` and ``TransformerMixin``.
    """

    def __init__(self, strategy: str = "mean", **kwargs) -> None:
        """Initialise the FittedImputer.

        Parameters
        ----------
        strategy : str
            Imputation strategy name.
        **kwargs
            Forwarded to MissinglyImputer.
        """
        # Import here to avoid circular import at module load time
        from .transformer import MissinglyImputer
        self.strategy = strategy
        self._imputer = MissinglyImputer(strategy=strategy, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """Return True if fit() has been called."""
        return self._imputer._is_fitted

    def fit(self, X: pd.DataFrame, y=None) -> "FittedImputer":
        """Learn imputation parameters from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training data that may contain missing values.  Only this
            data is used to compute imputation statistics (means, medians,
            KNN distances, etc.).
        y : ignored

        Returns
        -------
        self
        """
        self._imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Data to impute.  Must have the same columns as the training
            DataFrame passed to :meth:`fit`.

        Returns
        -------
        pd.DataFrame
            Fully-imputed copy of *X*.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FittedImputer.transform() called before fit(). "
                "Call fit(X_train) first."
            )
        return self._imputer.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit on *X* and return the imputed version of *X*.

        Equivalent to calling ``fit(X).transform(X)``.

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        pd.DataFrame
        """
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        fitted = "fitted" if self.is_fitted else "unfitted"
        return f"FittedImputer(strategy={self.strategy!r}, {fitted})"


def make_imputer(strategy: str = "mean", **kwargs) -> FittedImputer:
    """Factory function that returns an unfitted :class:`FittedImputer`.

    Use this instead of calling the stateless ``impute_*`` functions when
    you need to fit on training data and transform test data separately
    (i.e. to avoid data leakage).

    Parameters
    ----------
    strategy : str
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"knn"``,
        ``"mice"``, ``"rf"``, ``"gb"``.
    **kwargs
        Forwarded to :class:`FittedImputer` and ultimately to
        :class:`~missingly.transformer.MissinglyImputer`.

    Returns
    -------
    FittedImputer

    Example
    -------
    >>> from missingly.impute import make_imputer
    >>> imp = make_imputer('knn', n_neighbors=3)
    >>> imp.fit(X_train)
    >>> X_test_clean = imp.transform(X_test)
    """
    return FittedImputer(strategy=strategy, **kwargs)
