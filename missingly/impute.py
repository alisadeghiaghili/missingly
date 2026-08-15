"""Imputation utilities for missing data.

This module provides two layers of imputation API:

1. **Stateless functions** (``impute_mean``, ``impute_median``, etc.)
   Accept a DataFrame, fit on that same DataFrame, and return a filled
   copy.  Convenient for exploration, but they re-fit on every call so
   they must **not** be used across train/test splits (data leakage).

2. **FittedImputer** (``make_imputer`` factory)
   A lightweight ``fit`` / ``transform`` / ``fit_transform`` wrapper
   that stores per-column fill values (mean, median, or mode) computed
   on training data and applies them to unseen data.  Supports only
   ``{mean, median, mode}`` — strategies whose fit-state is a simple
   scalar per column.  For model-based strategies (knn, mice, rf, gb)
   use :class:`~missingly.transformer.MissinglyImputer` instead.

Key design decisions
--------------------
* Python ``None`` in object-dtype columns is normalised to ``np.nan``
  before any sklearn estimator sees the data.
* Numeric columns in ML-based imputers use the provided regressor.
* Categorical columns use a classifier, avoiding the error of treating
  category codes as continuous values.
* ``GradientBoostingRegressor`` / ``GradientBoostingClassifier`` do not
  accept NaN in feature matrices.  Any remaining NaN in the feature
  side is filled with column means computed from the training rows.

Error handling
--------------
All public functions validate their inputs eagerly and raise specific
:mod:`missingly.exceptions` exceptions on failure. No exception is swallowed
silently. When ``strict_mode=True`` (see :mod:`missingly.config` or the
global :data:`missingly.config.strict_mode` setting), ``impute_logreg``,
``impute_polyreg``, ``impute_polr``, ``impute_rf``, and ``impute_gb`` raise
:class:`~missingly.exceptions.ImputationError` rather than falling back to a
column mean or mode.

Large-data warnings
-------------------
Functions that are O(n²) or slow on large DataFrames emit a
``UserWarning`` when the input exceeds the configured
``large_df_threshold`` (default 50 000 rows; configure via
:attr:`missingly.config.large_df_threshold`).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.exceptions import NotFittedError

from missingly.exceptions import (
    ConfigurationError,
    ImputationError,
    InsufficientDataError,
    InvalidStrategyError,
)
from missingly.mi_contracts import ImputationPlan, ImputationResult, MIData, MissingnessSchema
from missingly._validation import (
    validate_dataframe,
    validate_positive_int,
    validate_strategy,
)
from missingly.config import config as _config

logger = logging.getLogger(__name__)

_SUPPORTED_STRATEGIES = frozenset({
    "mean", "median", "mode", "knn", "mice", "rf", "gb",
    "pmm", "logreg", "polyreg", "polr",
})
_SUPPORTED_KNN_METRICS = frozenset({"euclidean", "mixed"})
# Strategies whose fit-state is a per-column scalar — the only ones FittedImputer supports.
_SIMPLE_FIT_STRATEGIES = frozenset({"mean", "median", "mode"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_strict_mode(strict_mode: Optional[bool]) -> bool:
    """Resolve and validate a strict-mode override or package default.

    Parameters
    ----------
    strict_mode : bool or None
        Per-call strict-mode override. ``None`` selects
        :attr:`missingly.config.strict_mode`.

    Returns
    -------
    bool
        The validated strict-mode value used for the current imputation call.

    Raises
    ------
    ConfigurationError
        If the override or selected package default is not exactly ``True``
        or ``False``. Integers such as ``1`` are rejected.

    Examples
    --------
    >>> _resolve_strict_mode(False)
    False
    """
    if strict_mode is None:
        value = _config.strict_mode
        parameter = "missingly.config.strict_mode"
    else:
        value = strict_mode
        parameter = "strict_mode"

    if type(value) is not bool:
        raise ConfigurationError(
            param=parameter,
            got=value,
            reason="must be exactly True or False",
        )
    return value


def _warn_if_large(df: pd.DataFrame, method_name: str) -> None:
    """Emit a UserWarning when *df* exceeds the large-DataFrame threshold."""
    if len(df) > _config.large_df_threshold:
        warnings.warn(
            f"{method_name}: DataFrame has {len(df):,} rows which may result in "
            f"very long runtimes or high memory usage. "
            f"Consider sampling your data or using impute_mean / impute_median "
            f"for large datasets.",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_knn_heavy_categorical(df: pd.DataFrame, n_neighbors: int) -> None:
    """Warn when KNN is applied to a DataFrame dominated by categorical columns."""
    n_cat = len(df.select_dtypes(exclude=[np.number]).columns)
    n_num = len(df.select_dtypes(include=[np.number]).columns)
    if n_cat > n_num and n_neighbors > _config.knn_cat_neighbors_threshold:
        warnings.warn(
            f"KNN imputation may be unreliable: the DataFrame has {n_cat} categorical "
            f"column(s) but only {n_num} numeric column(s), and n_neighbors={n_neighbors} "
            f"(> {_config.knn_cat_neighbors_threshold}). "
            "Euclidean distance over ordinal-encoded categoricals is not statistically "
            "sound. Consider impute_rf(), impute_gb(), or impute_mice() for "
            "heavy-categorical datasets.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Python ``None`` with ``np.nan`` in object columns."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].where(df[obj_cols].notna(), other=np.nan)
    return df


def _fill_nan_with_col_means(X: np.ndarray) -> np.ndarray:
    """Replace NaN entries in a 2-D array with column means.

    Used by GradientBoosting imputers which do not natively accept NaN in
    their feature matrices.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Array that may contain NaN values.

    Returns
    -------
    np.ndarray
        Copy of *X* with NaN replaced by per-column means.
        Columns that are entirely NaN are filled with ``0.0``.
    """
    X = X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X


def _encode_features(X: np.ndarray) -> np.ndarray:
    """Encode a 2-D object/mixed array to float, replacing non-numeric with ordinal codes.

    Each column is processed independently:
    * Numeric-castable columns are kept as-is after float conversion.
    * String/object columns are label-encoded to integer codes (0-based),
      with NaN preserved as ``np.nan``.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray of float64
        Encoded matrix safe to pass to sklearn estimators that require
        a numeric input.
    """
    n_samples, n_features = X.shape
    X_out = np.empty((n_samples, n_features), dtype=np.float64)
    for j in range(n_features):
        col = X[:, j]
        try:
            X_out[:, j] = col.astype(np.float64)
        except (ValueError, TypeError):
            # Object/string column — label-encode, NaN stays NaN
            unique_vals = [v for v in col if v is not None and not _is_nan_scalar(v)]
            unique_vals = sorted(set(unique_vals), key=str)
            mapping = {v: float(i) for i, v in enumerate(unique_vals)}
            X_out[:, j] = np.array(
                [mapping.get(v, np.nan) if not _is_nan_scalar(v) else np.nan for v in col],
                dtype=np.float64,
            )
    return X_out


def _encode_feature_pair(
    X_observed: np.ndarray,
    X_missing: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode observed and incomplete feature rows in one shared code space.

    Conditional categorical imputers fit on ``X_observed`` and predict on
    ``X_missing``.  Encoding the two arrays independently changes the numeric
    coordinate assigned to a category that occurs in only one subset.  This
    helper derives every per-column mapping once from their union, then splits
    the encoded rows back into their original groups.

    Parameters
    ----------
    X_observed : np.ndarray
        Raw feature rows associated with observed target values, with shape
        ``(n_observed, n_features)``.
    X_missing : np.ndarray
        Raw feature rows associated with missing target values, with shape
        ``(n_missing, n_features)``.

    Returns
    -------
    tuple of np.ndarray
        ``(X_observed_encoded, X_missing_encoded)`` in one shared float64
        coordinate system.

    Raises
    ------
    ValueError
        If the arrays do not have the same number of feature columns.

    Examples
    --------
    >>> observed = np.array([["a"], ["e"]], dtype=object)
    >>> missing = np.array([["e"]], dtype=object)
    >>> X_observed, X_missing = _encode_feature_pair(observed, missing)
    >>> X_observed[1, 0] == X_missing[0, 0]
    np.True_
    """
    if X_observed.ndim != 2 or X_missing.ndim != 2:
        raise ValueError("Feature arrays must both be two-dimensional.")
    if X_observed.shape[1] != X_missing.shape[1]:
        raise ValueError("Feature arrays must have the same number of columns.")

    n_observed = X_observed.shape[0]
    encoded = _encode_features(np.concatenate([X_observed, X_missing], axis=0))
    return encoded[:n_observed], encoded[n_observed:]


def _ordinal_target_categories(series: pd.Series) -> List:
    """Return the category order used by an ordinal conditional model.

    Parameters
    ----------
    series : pd.Series
        Observed and missing values of one ordinal target.  An ordered pandas
        categorical supplies its declared substantive order; all other dtypes
        use deterministic lexical ordering for backward compatibility.

    Returns
    -------
    list
        Ordered target categories, excluding missing values.

    Examples
    --------
    >>> values = pd.Series(pd.Categorical(["low", "high"], categories=["low", "high"], ordered=True))
    >>> _ordinal_target_categories(values)
    ['low', 'high']
    """
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return series.cat.categories.tolist()
    return sorted(series.dropna().unique(), key=str)


def _is_nan_scalar(v) -> bool:
    """Return whether a scalar represents a missing value.

    Parameters
    ----------
    v : object
        Scalar candidate to inspect.

    Returns
    -------
    bool
        ``True`` for Python ``None``, ``numpy.nan``, ``pandas.NA``, and other
        scalar values recognised as missing by :func:`pandas.isna`.

    Examples
    --------
    >>> _is_nan_scalar(pd.NA)
    True
    >>> _is_nan_scalar("observed")
    False
    """
    try:
        is_missing = pd.isna(v)
        return bool(is_missing) if np.isscalar(is_missing) else False
    except (TypeError, ValueError):
        return False


def _fill_feature_matrix(
    X_obs: np.ndarray,
    X_miss: np.ndarray,
) -> tuple:
    """Fill NaN in feature matrices with column means derived from X_obs.

    Parameters
    ----------
    X_obs : np.ndarray
        Feature matrix for observed rows.
    X_miss : np.ndarray
        Feature matrix for rows to predict.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(X_obs_clean, X_miss_clean)`` with no NaN values.
    """
    X_obs_clean = X_obs.copy()
    X_miss_clean = X_miss.copy()
    for j in range(X_obs_clean.shape[1]):
        col_mean = np.nanmean(X_obs_clean[:, j])
        if np.isnan(col_mean):
            col_mean = 0.0
        X_obs_clean[np.isnan(X_obs_clean[:, j]), j] = col_mean
        X_miss_clean[np.isnan(X_miss_clean[:, j]), j] = col_mean
    return X_obs_clean, X_miss_clean


def _restore_categorical_dtype(
    series: pd.Series,
    orig_dtype,
) -> pd.Series:
    """Restore a Series to its original categorical or plain dtype.

    Safer than ``series.astype(orig_dtype)`` for CategoricalDtype because
    it reconstructs the Categorical explicitly, preserving the ``ordered``
    flag without crashing when new values appear.

    Parameters
    ----------
    series : pd.Series
        Imputed series (dtype may be object/string after SimpleImputer).
    orig_dtype : dtype
        Original dtype of the column before imputation.

    Returns
    -------
    pd.Series
        Series with dtype restored as closely as possible.
    """
    if hasattr(orig_dtype, "categories"):
        ordered = getattr(orig_dtype, "ordered", False)
        return pd.Series(
            pd.Categorical(series, categories=orig_dtype.categories, ordered=ordered),
            index=series.index,
        )
    try:
        return series.astype(orig_dtype)
    except (ValueError, TypeError):
        return series


def _split_encode(
    df: pd.DataFrame,
):
    """Ordinal-encode categorical columns; return parts needed to reconstruct.

    Returns
    -------
    tuple
        ``(df_work, cat_cols, num_cols, encoder, cat_dtypes)``
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_dtypes = {c: df[c].dtype for c in cat_cols}

    if not cat_cols:
        return df, cat_cols, num_cols, None, cat_dtypes

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    categorical_values = df.loc[:, cat_cols].astype(object)
    categorical_values = categorical_values.where(categorical_values.notna(), np.nan)
    cat_encoded = encoder.fit_transform(categorical_values)
    df_work = df[num_cols].copy()
    for i, col in enumerate(cat_cols):
        df_work[col] = cat_encoded[:, i]
    return df_work, cat_cols, num_cols, encoder, cat_dtypes


def _decode(
    df_imputed: pd.DataFrame,
    cat_cols: list,
    encoder,
    cat_dtypes: dict,
) -> pd.DataFrame:
    """Inverse-transform ordinal-encoded columns back to their original dtype."""
    if not cat_cols or encoder is None:
        return df_imputed

    cat_array = df_imputed[cat_cols].values.astype(float)
    result = df_imputed.copy()

    for i, col in enumerate(cat_cols):
        n_cats = len(encoder.categories_[i])
        col_vals = cat_array[:, i].copy()

        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            valid = col_vals[~nan_mask]
            if len(valid) > 0:
                fallback = float(
                    np.bincount(
                        np.round(np.clip(valid, 0, n_cats - 1)).astype(int)
                    ).argmax()
                )
            else:
                fallback = 0.0
            col_vals[nan_mask] = fallback

        col_vals = np.clip(np.round(col_vals), 0, n_cats - 1).astype(int)
        decoded_col = encoder.categories_[i][col_vals]

        result[col] = _restore_categorical_dtype(
            pd.Series(decoded_col, index=result.index),
            cat_dtypes[col],
        )

    return result


def _impute_column_by_column(
    df: pd.DataFrame,
    regressor,
    classifier,
    random_state: int,
    strategy_name: str,
    max_iter: int = 1,
    fill_feature_nan: bool = False,
    strict_mode: Optional[bool] = None,
) -> pd.DataFrame:
    """Column-by-column imputation used by RF and GB imputers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (already normalised, no Python None).
    regressor : sklearn estimator
        Estimator used for numeric columns.
    classifier : sklearn estimator
        Estimator used for categorical columns.
    random_state : int
        Random seed (informational; estimators are already initialised).
    strategy_name : str
        Human-readable strategy name used in exception messages.
    max_iter : int, default 1
        Number of imputation passes.
    fill_feature_nan : bool, default False
        If True, fill NaN in the feature matrix with column means before
        fitting.  Required for GradientBoosting which rejects NaN inputs.
    strict_mode : bool or None, default None
        If True, raise :class:`~missingly.exceptions.ImputationError` on
        estimator failure instead of falling back to the column mean/mode.
        If None, falls back to :attr:`missingly.config.strict_mode`. Values
        must be exactly ``True`` or ``False``.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.

    Raises
    ------
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.
    """
    effective_strict = _resolve_strict_mode(strict_mode)

    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)
    all_cols = list(df_work.columns)
    df_result = df_work.copy()

    for _ in range(max_iter):
        for col in all_cols:
            missing_mask = df_work[col].isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in all_cols if c != col]
            X_full = df_result[feature_cols].values
            y_full = df_result[col].values

            observed_mask = ~np.isnan(y_full)
            n_observed = observed_mask.sum()

            if n_observed == 0:
                logger.info(
                    "Column %r has no observed values; skipping imputation.", col
                )
                raise InsufficientDataError(
                    column=col, n_observed=0, n_required=1
                )

            X_train = X_full[observed_mask]
            y_train = y_full[observed_mask]
            X_pred = X_full[missing_mask.values]

            if fill_feature_nan:
                X_train = _fill_nan_with_col_means(X_train)
                X_pred = _fill_nan_with_col_means(X_pred)

            est = classifier if col in cat_cols else regressor

            try:
                est.fit(X_train, y_train)
                preds = est.predict(X_pred)
            except (ValueError, np.linalg.LinAlgError, NotFittedError) as exc:
                if effective_strict:
                    raise ImputationError(
                        column=col, strategy=strategy_name, original=exc
                    ) from exc
                if col in cat_cols:
                    valid_codes = y_train[~np.isnan(y_train)].astype(int)
                    valid_codes = valid_codes[valid_codes >= 0]
                    fallback = float(np.bincount(valid_codes).argmax()) if len(valid_codes) > 0 else 0.0
                    logger.info(
                        "Categorical fallback for %r (strategy=%r): mode code %d. "
                        "Set strict_mode=True to raise instead.",
                        col, strategy_name, int(fallback),
                    )
                else:
                    fallback = float(np.nanmean(y_train))
                    logger.info(
                        "Numeric fallback for %r (strategy=%r): mean %.4f. "
                        "Set strict_mode=True to raise instead.",
                        col, strategy_name, fallback,
                    )
                preds = fallback * np.ones(X_pred.shape[0])

            df_result.loc[missing_mask, col] = preds

    return _decode(df_result, cat_cols, encoder, cat_dtypes)


# ---------------------------------------------------------------------------
# MICE history tracking helpers
# ---------------------------------------------------------------------------

class _HistoryTrackingEstimator(BaseEstimator, RegressorMixin):
    """Thin wrapper around a base estimator that records per-predict means.

    Used internally by :func:`impute_mice` when ``return_history=True``.
    Each call to ``predict`` appends the mean of the predictions to
    ``self.history_`` for the given column key.

    Parameters
    ----------
    base_estimator : sklearn estimator
        The underlying regression estimator.
    history : dict
        Shared mutable dict mapping column-name → list of floats.
    col_name : str
        Key into *history* for this column's trace.
    """

    def __init__(self, base_estimator, history: dict, col_name: str) -> None:
        self.base_estimator = base_estimator
        self.history = history
        self.col_name = col_name

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        return self

    def predict(self, X):
        preds = self.base_estimator_.predict(X)
        self.history.setdefault(self.col_name, []).append(float(np.mean(preds)))
        return preds


def _run_mice_with_history(
    df_work: pd.DataFrame,
    missing_mask_df: pd.DataFrame,
    base_estimator,
    max_iter: int,
    seed: int,
    use_sample_posterior: bool,
    visit_sequence: Optional[List[str]] = None,
    predictor_matrix: Optional[pd.DataFrame] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Run MICE manually iteration-by-iteration to capture per-iteration means.

    sklearn's ``IterativeImputer`` does not expose intermediate states after
    each iteration.  This function replicates the imputation order ("roman"
    = left-to-right) manually so that we can snapshot the imputed means at
    the end of every full sweep.

    Parameters
    ----------
    df_work : pd.DataFrame
        Ordinal-encoded, numeric-only working DataFrame.
    missing_mask_df : pd.DataFrame
        Boolean DataFrame; True where a value was originally missing.
    base_estimator : sklearn estimator
        Regressor to use for each column.
    max_iter : int
        Number of full sweeps.
    seed : int
        Random seed passed to the estimator where applicable.
    use_sample_posterior : bool
        Whether the estimator supports ``sample_posterior`` (BayesianRidge).
    visit_sequence : list of str, optional
        Ordered incomplete target columns to update in each sweep. ``None``
        preserves the default left-to-right order.
    predictor_matrix : pandas.DataFrame, optional
        Square validated 0/1 or boolean matrix whose target rows select feature
        columns. ``None`` uses every non-target column.
    bounds : dict of str to tuple of float, optional
        Inclusive lower and upper bounds applied only to newly imputed numeric
        target values.

    Returns
    -------
    imputed_array : np.ndarray
        Final imputed values, same shape as ``df_work``.
    history : Dict[str, List[float]]
        ``{col: [mean_iter1, mean_iter2, ...]}``.  Only columns that had
        missing values are included.
    """
    rng = np.random.default_rng(seed)
    cols = list(df_work.columns)
    # Initialise missing cells with column means (same as sklearn default)
    X = df_work.values.astype(float).copy()
    for j in range(X.shape[1]):
        col_missing = np.isnan(X[:, j])
        if col_missing.any():
            col_mean = np.nanmean(X[:, j])
            X[col_missing, j] = col_mean if not np.isnan(col_mean) else 0.0

    history: Dict[str, List[float]] = {}

    visit_cols = visit_sequence if visit_sequence is not None else cols
    for _ in range(max_iter):
        for col in visit_cols:
            j = cols.index(col)
            orig_missing = missing_mask_df[col].values
            if not orig_missing.any():
                continue

            if predictor_matrix is None:
                feature_idx = [k for k in range(len(cols)) if k != j]
            else:
                feature_idx = np.flatnonzero(
                    predictor_matrix.loc[col, cols].to_numpy(dtype=bool)
                ).tolist()
            X_train = X[~orig_missing][:, feature_idx]
            y_train = X[~orig_missing, j]
            X_pred_rows = X[orig_missing][:, feature_idx]

            est = clone(base_estimator)
            # Seed estimators that accept random_state
            if hasattr(est, "random_state"):
                est.set_params(random_state=int(rng.integers(0, 2**31)))
            try:
                est.fit(X_train, y_train)
                if use_sample_posterior:
                    predictive_mean, predictive_std = est.predict(
                        X_pred_rows, return_std=True
                    )
                    preds = rng.normal(predictive_mean, predictive_std)
                else:
                    preds = est.predict(X_pred_rows)
            except (ValueError, np.linalg.LinAlgError, NotFittedError):
                preds = np.full(orig_missing.sum(), float(np.mean(y_train)))

            if bounds is not None and col in bounds:
                lower, upper = bounds[col]
                preds = np.clip(preds, lower, upper)

            X[orig_missing, j] = preds
            history.setdefault(col, []).append(float(np.mean(preds)))

    return X, history


# ---------------------------------------------------------------------------
# Public imputation functions
# ---------------------------------------------------------------------------

def impute_mean(
    df: pd.DataFrame,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Impute missing values with column means.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_only : bool, default False
        If True, only numeric columns are imputed; non-numeric columns
        are left unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by column means.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    >>> impute_mean(df)
         a
    0  1.0
    1  2.0
    2  3.0
    """
    validate_dataframe(df, param="df")
    df = _normalize_missing(df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    result = df.copy()
    if len(num_cols):
        imputer = SimpleImputer(strategy="mean")
        result[num_cols] = imputer.fit_transform(df[num_cols])

    if not numeric_only and len(cat_cols):
        mode_imputer = SimpleImputer(strategy="most_frequent")
        imputed = mode_imputer.fit_transform(df[cat_cols])
        for i, col in enumerate(cat_cols):
            filled = pd.Series(imputed[:, i], index=df.index)
            result[col] = _restore_categorical_dtype(filled, df[col].dtype)

    return result


def impute_median(
    df: pd.DataFrame,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Impute missing values with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_only : bool, default False
        If True, only numeric columns are imputed.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by column medians.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    >>> impute_median(df)
         a
    0  1.0
    1  2.0
    2  3.0
    """
    validate_dataframe(df, param="df")
    df = _normalize_missing(df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    result = df.copy()
    if len(num_cols):
        imputer = SimpleImputer(strategy="median")
        result[num_cols] = imputer.fit_transform(df[num_cols])

    if not numeric_only and len(cat_cols):
        mode_imputer = SimpleImputer(strategy="most_frequent")
        imputed = mode_imputer.fit_transform(df[cat_cols])
        for i, col in enumerate(cat_cols):
            filled = pd.Series(imputed[:, i], index=df.index)
            result[col] = _restore_categorical_dtype(filled, df[col].dtype)

    return result


def impute_mode(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Impute missing values with column modes (most frequent values).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by column modes.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"city": ["Berlin", np.nan, "Berlin"]})
    >>> impute_mode(df)["city"].tolist()
    ['Berlin', 'Berlin', 'Berlin']
    """
    validate_dataframe(df, param="df")
    df = _normalize_missing(df)
    imputer = SimpleImputer(strategy="most_frequent")
    imputed = imputer.fit_transform(df)
    result = pd.DataFrame(imputed, index=df.index, columns=df.columns)
    for col in df.columns:
        result[col] = _restore_categorical_dtype(result[col], df[col].dtype)
    return result


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbours.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_neighbors : int, default 5
        Number of nearest neighbours to use.
    metric : {"euclidean", "mixed"}, default "euclidean"
        Distance metric to use.

        * ``"euclidean"`` — Ordinal-encode categorical columns and apply
          sklearn's :class:`~sklearn.impute.KNNImputer` with Euclidean
          distance.  Fast and works well for purely numeric data.
        * ``"mixed"`` — Use Gower distance which handles numeric and
          categorical columns natively.  Statistically sounder for
          heavy-categorical datasets, but O(n²).  Not recommended for
          ``n > 10 000``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values imputed via KNN.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    InvalidStrategyError
        If *metric* is not one of the supported values.
    TypeError
        If *n_neighbors* is not a positive integer.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"age": [25.0, np.nan, 35.0, 40.0]})
    >>> result = impute_knn(df, n_neighbors=2)
    >>> round(float(result.loc[1, "age"]), 1)
    33.3
    """
    validate_dataframe(df, param="df")
    validate_positive_int(n_neighbors, param="n_neighbors")
    validate_strategy(
        metric,
        _SUPPORTED_KNN_METRICS,
        param="metric",
        example="euclidean",
    )

    _warn_if_large(df, "impute_knn")
    df = _normalize_missing(df)

    if metric == "mixed":
        return _impute_knn_gower(df, n_neighbors=n_neighbors)

    _warn_if_knn_heavy_categorical(df, n_neighbors)
    df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(
        imputed_array, index=df.index, columns=df_work.columns
    )
    return _decode(df_imputed, cat_cols, encoder, cat_dtypes)


def impute_mice(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    estimator=None,
    n_imputations: int = 1,
    return_history: bool = False,
    return_result: bool = False,
    visit_sequence: Optional[List[str]] = None,
    predictor_matrix: Optional[pd.DataFrame] = None,
    where: Optional[pd.DataFrame] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Union[
    pd.DataFrame,
    List[pd.DataFrame],
    Tuple[pd.DataFrame, Dict[str, List[float]]],
    Tuple[List[pd.DataFrame], List[Dict[str, List[float]]]],
    ImputationResult,
]:
    """Impute missing values using MICE (Multiple Imputation by Chained Equations).

    When ``n_imputations > 1`` runs independent chains each with a distinct
    random seed so the returned DataFrames differ from one another.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE imputation iterations per chain.
    random_state : int, default 0
        Base random seed.
    estimator : sklearn estimator or None, default None
        Regression estimator used per column.  Defaults to
        ``BayesianRidge()`` with posterior sampling enabled.
    n_imputations : int, default 1
        Number of independently imputed DataFrames to generate.
        Returns a single DataFrame when 1, a list when > 1.
    return_history : bool, default False
        If True, also return per-iteration imputed means for each variable
        that had missing values.  Used by :func:`~missingly.diagnostics.mice_convergence`
        for trace plots and Gelman-Rubin R-hat computation.

        When ``n_imputations == 1`` and ``return_history=True`` the return
        value is ``(imputed_df, history)`` where *history* is
        ``Dict[str, List[float]]`` mapping column name to a list of
        per-iteration mean imputed values.

        When ``n_imputations > 1`` and ``return_history=True`` the return
        value is ``(list_of_dfs, list_of_histories)``.

        With the default Bayesian ridge estimator, each chain uses posterior
        predictive draws. Histories therefore preserve between-chain
        stochasticity and are suitable for diagnostic use; they are still not
        proof that the imputation model or missing-data assumptions are valid.
    return_result : bool, default False
        Return an immutable :class:`~missingly.mi_contracts.ImputationResult`
        containing the original mask, chain seeds, completed datasets, and
        per-iteration histories. This opt-in mode is mutually exclusive with
        ``return_history`` because the result object already includes history.
    visit_sequence : list of str, optional
        Exact ordered set of columns with original missing values to update in
        every FCS sweep. ``None`` uses the legacy left-to-right order.
    predictor_matrix : pandas.DataFrame, optional
        Square boolean or 0/1 matrix indexed and columned by ``df.columns``.
        A ``True`` / ``1`` entry at ``[target, predictor]`` enables that
        predictor in the target's conditional model. The diagonal must be zero.
    where : pandas.DataFrame, optional
        Boolean mask aligned exactly to ``df``. ``True`` permits imputation of
        an originally missing cell; ``False`` retains it as missing. Over-
        imputation of originally observed cells is rejected.
    bounds : dict of str to tuple of float, optional
        Inclusive numeric bounds per target column. Bounds are applied only to
        newly imputed values and must use finite ascending limits.

    Returns
    -------
    pd.DataFrame, list of pd.DataFrame, or ImputationResult
        Imputed copy (or list of *m* copies) of *df*.  When
        ``return_history=True`` a tuple is returned instead — see above.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty or *n_imputations* < 1.
    InsufficientDataError
        If an enabled target column has no observed values from which an
        imputation model can learn.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
    >>> result = impute_mice(df, max_iter=2, random_state=0)
    >>> bool(result["a"].isna().any())
    False

    >>> imputed, history = impute_mice(df, max_iter=5, return_history=True)
    >>> len(history["a"])  # one entry per iteration
    5

    >>> typed = impute_mice(df, n_imputations=2, return_result=True)
    >>> typed.data.n_imputations
    2
    """
    validate_dataframe(df, param="df")
    validate_positive_int(n_imputations, param="n_imputations")
    validate_positive_int(max_iter, param="max_iter")
    if return_history and return_result:
        raise ValueError("return_result cannot be combined with return_history")

    _warn_if_large(df, "impute_mice")
    df_norm = _normalize_missing(df)
    original_missing_mask = df_norm.isna()
    if where is None:
        imputation_mask = original_missing_mask.copy()
    else:
        if not isinstance(where, pd.DataFrame):
            raise TypeError("where must be a pandas DataFrame")
        if not where.index.equals(df_norm.index) or tuple(where.columns) != tuple(df_norm.columns):
            raise ValueError("where must align exactly with df index and columns")
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in where.dtypes):
            raise TypeError("where must contain only boolean columns")
        if (where & ~original_missing_mask).any(axis=None):
            raise ValueError("where cannot request imputation of observed cells")
        imputation_mask = original_missing_mask & where
    retained_missing_mask = original_missing_mask & ~imputation_mask
    missing_targets = [column for column in df_norm.columns if imputation_mask[column].any()]
    if visit_sequence is not None:
        if not isinstance(visit_sequence, list) or not all(
            isinstance(column, str) for column in visit_sequence
        ):
            raise TypeError("visit_sequence must be a list of column names")
        unknown = sorted(set(visit_sequence) - set(df_norm.columns))
        if unknown:
            raise ValueError(f"visit_sequence contains unknown columns: {unknown}")
        if len(visit_sequence) != len(set(visit_sequence)) or set(visit_sequence) != set(missing_targets):
            raise ValueError("visit_sequence must contain every missing target exactly once")
    if predictor_matrix is not None:
        if not isinstance(predictor_matrix, pd.DataFrame):
            raise TypeError("predictor_matrix must be a pandas DataFrame")
        expected_labels = tuple(df_norm.columns)
        if tuple(predictor_matrix.index) != expected_labels or tuple(predictor_matrix.columns) != expected_labels:
            raise ValueError("predictor_matrix labels must exactly match df.columns")
        values = predictor_matrix.to_numpy()
        if pd.isna(values).any() or not np.isin(values, [False, True, 0, 1]).all():
            raise ValueError("predictor_matrix values must be boolean or 0/1")
        if np.any(np.diag(values.astype(bool))):
            raise ValueError("predictor_matrix diagonal must contain only zero values")
        predictor_matrix = predictor_matrix.astype(bool)
    if bounds is not None:
        if not isinstance(bounds, dict):
            raise TypeError("bounds must be a dictionary of column intervals")
        unknown_bounds = sorted(set(bounds) - set(df_norm.columns))
        if unknown_bounds:
            raise ValueError(f"bounds contains unknown columns: {unknown_bounds}")
        for column, interval in bounds.items():
            if not pd.api.types.is_numeric_dtype(df_norm[column]):
                raise ValueError(f"bounds column {column!r} must be numeric")
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("each bounds interval must be a (lower, upper) tuple")
            lower, upper = interval
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in interval):
                raise ValueError("bounds must contain finite numeric limits")
            if lower > upper:
                raise ValueError("bounds lower limit must not exceed upper limit")

    for column in missing_targets:
        n_observed = int((~original_missing_mask[column]).sum())
        if n_observed == 0:
            raise InsufficientDataError(
                column=column,
                n_observed=0,
                n_required=1,
            )

    def _single_chain(
        seed: int,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, List[float]]]]:
        df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df_norm)
        capture_history = return_history or return_result

        if return_history or return_result or visit_sequence is not None or predictor_matrix is not None or where is not None or bounds is not None:
            # Use our manual per-iteration loop so we can capture history.
            missing_mask_df = imputation_mask.loc[:, df_work.columns]
            if estimator is None:
                base_est = BayesianRidge()
                use_posterior = True
            else:
                base_est = estimator
                use_posterior = False

            imputed_array, history = _run_mice_with_history(
                df_work=df_work,
                missing_mask_df=missing_mask_df,
                base_estimator=base_est,
                max_iter=max_iter,
                seed=seed,
                use_sample_posterior=use_posterior,
                visit_sequence=visit_sequence,
                predictor_matrix=predictor_matrix,
                bounds=bounds,
            )
            df_imputed = pd.DataFrame(
                imputed_array, index=df_norm.index, columns=df_work.columns
            )
            result = _decode(df_imputed, cat_cols, encoder, cat_dtypes)
            for col in cat_cols:
                if result[col].isna().any():
                    observed = df_norm[col].dropna()
                    if len(observed) > 0:
                        result[col] = result[col].fillna(observed.mode().iloc[0])
            result = result.mask(retained_missing_mask.loc[:, result.columns])
            if capture_history:
                return result, history
            return result

        # Fast path: delegate to sklearn's IterativeImputer (no history).
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
        result = _decode(df_imputed, cat_cols, encoder, cat_dtypes)
        for col in cat_cols:
            if result[col].isna().any():
                observed = df_norm[col].dropna()
                if len(observed) > 0:
                    fallback = observed.mode().iloc[0]
                    result[col] = result[col].fillna(fallback)
        return result

    if return_result:
        results = [_single_chain(random_state + index) for index in range(n_imputations)]
        frames, histories = zip(*results)
        schema = MissingnessSchema.from_dataframe(df)
        plan = ImputationPlan(
            methods={column: "mice" for column in df.columns},
            n_imputations=n_imputations,
            max_iter=max_iter,
            seed_sequence=tuple(random_state + index for index in range(n_imputations)),
            provenance={"engine": "impute_mice", "history": "per_iteration_mean"},
        )
        data = MIData(df.copy(deep=True), schema, plan, tuple(frames))
        contract_histories = tuple(
            {column: tuple(float(value) for value in trace) for column, trace in history.items()}
            for history in histories
        )
        return ImputationResult(
            data=data,
            histories=contract_histories,
            warnings=(
                "Convergence has not been assessed. Run mice_convergence on the "
                "returned histories before interpreting chain stability.",
            ),
            validation={
                "structural_validation_passed": True,
                "convergence_assessed": False,
            },
        )

    if n_imputations == 1:
        return _single_chain(random_state)

    results = [_single_chain(random_state + i) for i in range(n_imputations)]
    if return_history:
        dfs, histories = zip(*results)
        return list(dfs), list(histories)
    return list(results)


def impute_pmm(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    n_nearest_donors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using Predictive Mean Matching (PMM).

    PMM is the most commonly used method in R's mice package. It works by:

    1. Fitting a predictive model for each variable with missing values.
    2. Using the model to predict values for missing observations.
    3. Finding the closest observed values (donors) to each predicted value.
    4. Randomly selecting one donor as the imputed value.

    This preserves the distribution of the observed data better than direct
    regression imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations per chain.
    random_state : int, default 0
        Random seed for reproducibility.
    n_nearest_donors : int, default 5
        Number of nearest observed values to consider as donors for each
        missing value. Larger values increase randomness.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    >>> result = impute_pmm(df, random_state=0)
    >>> bool(result["a"].isna().any())
    False

    Notes
    -----
    This is an experimental, single-dataset chained-equations approximation.
    It uses type-1 PMM matching: donor predictive means use the fitted
    regression coefficients, while incomplete-row means use a draw from the
    BayesianRidge coefficient posterior.  Use a validated multiple-imputation
    workflow and sensitivity analysis for publication-critical inference.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")
    validate_positive_int(n_nearest_donors, param="n_nearest_donors")

    _warn_if_large(df, "impute_pmm")
    df_norm = _normalize_missing(df)

    num_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_norm.select_dtypes(exclude=[np.number]).columns.tolist()

    if not num_cols:
        return impute_mode(df)

    rng = np.random.default_rng(random_state)
    result = df_norm.copy()
    missing_masks = {col: df_norm[col].isna() for col in num_cols}

    for _ in range(max_iter):
        for col in num_cols:
            missing_mask = missing_masks[col]
            if not missing_mask.any():
                continue

            feature_cols = [c for c in num_cols if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mean()
                continue

            observed_mask = ~missing_mask
            X_obs = result.loc[observed_mask, feature_cols].values.astype(np.float64)
            y_obs = result.loc[observed_mask, col].values.astype(np.float64)
            X_miss = result.loc[missing_mask, feature_cols].values.astype(np.float64)

            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs, X_miss)

            model = BayesianRidge()
            model.fit(X_obs_clean, y_obs)
            donor_scores = model.predict(X_obs_clean)
            coefficient_draw = rng.multivariate_normal(model.coef_, model.sigma_)
            recipient_scores = X_miss_clean @ coefficient_draw + model.intercept_

            # Type-1 PMM: match a posterior-drawn recipient score to a donor
            # score from the fitted model, then copy the donor's observed y.
            distances = np.abs(
                donor_scores[np.newaxis, :] - recipient_scores[:, np.newaxis]
            )
            k = min(n_nearest_donors, len(y_obs))
            donor_idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
            rand_col = rng.integers(0, k, size=len(recipient_scores))
            chosen_idx = donor_idx[np.arange(len(recipient_scores)), rand_col]
            imputed_vals = y_obs[chosen_idx]

            miss_indices = missing_mask[missing_mask].index
            result.loc[miss_indices, col] = imputed_vals

    for col in cat_cols:
        if result[col].isna().any():
            observed = result[col].dropna()
            if len(observed) > 0:
                result[col] = result[col].fillna(observed.mode().iloc[0])

    return result


def impute_logreg(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    strict_mode: Optional[bool] = None,
) -> pd.DataFrame:
    """Impute missing values using Logistic Regression (for binary variables).

    This experimental chained-equations approximation fits a logistic
    regression model for each binary variable with missing values, then samples
    from the predicted Bernoulli probability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations.
    random_state : int, default 0
        Random seed for reproducibility.
    strict_mode : bool or None, default None
        If ``True``, raise :class:`~missingly.exceptions.ImputationError`
        when a binary conditional model cannot fit or predict. If ``False``,
        fill with the observed mode and emit a :class:`UserWarning`. If
        ``None``, use :attr:`missingly.config.strict_mode`. Values must be
        exactly ``True`` or ``False``.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ImputationError
        If the conditional estimator fails and ``strict_mode=True``.
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 0.0, 1.0, 0.0]})
    >>> result = impute_logreg(df, random_state=0)
    >>> bool(result["a"].isna().any())
    False

    Notes
    -----
    This method is best suited for binary variables (0/1 or two-level
    categorical). For categorical variables with more than 2 levels,
    use ``impute_polyreg`` instead.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")

    effective_strict = _resolve_strict_mode(strict_mode)
    _warn_if_large(df, "impute_logreg")
    df_norm = _normalize_missing(df)

    rng = np.random.default_rng(random_state)
    result = df_norm.copy()

    binary_cols = [
        col for col in df_norm.columns
        if len(df_norm[col].dropna().unique()) == 2
    ]
    missing_masks = {col: df_norm[col].isna() for col in binary_cols}

    for _ in range(max_iter):
        for col in binary_cols:
            missing_mask = missing_masks[col]
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc, X_miss_enc = _encode_feature_pair(X_obs_raw, X_miss_raw)
            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs_enc, X_miss_enc)

            y_obs = result.loc[observed_mask, col].values

            model = LogisticRegression(random_state=random_state, max_iter=1000)
            try:
                model.fit(X_obs_clean, y_obs)
                y_pred_proba = model.predict_proba(X_miss_clean)[:, 1]
                imputed = np.where(
                    y_pred_proba > rng.random(len(y_pred_proba)),
                    model.classes_[1],
                    model.classes_[0],
                )
                result.loc[missing_mask, col] = imputed
            except (ValueError, np.linalg.LinAlgError, NotFittedError) as exc:
                if effective_strict:
                    raise ImputationError(
                        column=col, strategy="logreg", original=exc
                    ) from exc
                fallback = result[col].mode().iloc[0]
                warnings.warn(
                    f"impute_logreg: column {col!r} model failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"Falling back to mode={fallback!r}. Set strict_mode=True to raise.",
                    UserWarning,
                    stacklevel=2,
                )
                result.loc[missing_mask, col] = fallback

    return result


def impute_polyreg(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    strict_mode: Optional[bool] = None,
) -> pd.DataFrame:
    """Impute missing values using Multinomial Logistic Regression.

    This experimental chained-equations approximation fits a multinomial
    logistic regression model for each categorical variable with more than
    two levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations.
    random_state : int, default 0
        Random seed for reproducibility.
    strict_mode : bool or None, default None
        If ``True``, raise :class:`~missingly.exceptions.ImputationError`
        when a multinomial conditional model cannot fit or predict. If
        ``False``, fill with the observed mode and emit a
        :class:`UserWarning`. If ``None``, use
        :attr:`missingly.config.strict_mode`. Values must be exactly ``True``
        or ``False``.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ImputationError
        If the conditional estimator fails and ``strict_mode=True``.
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"color": ["red", np.nan, "blue", "green", "red"]})
    >>> result = impute_polyreg(df, random_state=0)
    >>> bool(result["color"].isna().any())
    False

    Notes
    -----
    Best suited for nominal categorical variables with 3+ levels.
    For binary variables use ``impute_logreg``.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")

    effective_strict = _resolve_strict_mode(strict_mode)
    _warn_if_large(df, "impute_polyreg")
    df_norm = _normalize_missing(df)

    result = df_norm.copy()

    cat_cols = [
        col for col in df_norm.columns
        if (
            len(df_norm[col].dropna().unique()) >= 3
            and not pd.api.types.is_numeric_dtype(df_norm[col])
        )
    ]
    missing_masks = {col: df_norm[col].isna() for col in cat_cols}

    for _ in range(max_iter):
        for col in cat_cols:
            missing_mask = missing_masks[col]
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc, X_miss_enc = _encode_feature_pair(X_obs_raw, X_miss_raw)
            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs_enc, X_miss_enc)

            y_obs = result.loc[observed_mask, col].values

            model = LogisticRegression(random_state=random_state, max_iter=1000)
            try:
                model.fit(X_obs_clean, y_obs)
                y_pred = model.predict(X_miss_clean)
                result.loc[missing_mask, col] = y_pred
            except (ValueError, np.linalg.LinAlgError, NotFittedError) as exc:
                if effective_strict:
                    raise ImputationError(
                        column=col, strategy="polyreg", original=exc
                    ) from exc
                fallback = result[col].mode().iloc[0]
                warnings.warn(
                    f"impute_polyreg: column {col!r} model failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"Falling back to mode={fallback!r}. "
                    "Set strict_mode=True to raise.",
                    UserWarning,
                    stacklevel=2,
                )
                result.loc[missing_mask, col] = fallback

    return result


def impute_polr(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
    strict_mode: Optional[bool] = None,
) -> pd.DataFrame:
    """Impute missing values using Ordinal Logistic Regression.

    This experimental chained-equations approximation fits an ordinal logistic
    regression model for each ordered categorical variable.
    Falls back to multinomial logistic regression when statsmodels is
    unavailable or the ordinal model fails to converge.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations.
    random_state : int, default 0
        Random seed for reproducibility.
    strict_mode : bool or None, default None
        If ``True``, raise :class:`~missingly.exceptions.ImputationError`
        when an ordinal or multinomial conditional model cannot fit or
        predict. If ``False``, use the documented fallback sequence and emit
        a :class:`UserWarning`. If ``None``, use
        :attr:`missingly.config.strict_mode`. Values must be exactly ``True``
        or ``False``.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ImputationError
        If a conditional estimator fails and ``strict_mode=True``.
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"grade": ["A", np.nan, "B", "C", "A"]})
    >>> result = impute_polr(df, random_state=0)
    >>> bool(result["grade"].isna().any())
    False

    Notes
    -----
    Best suited for ordinal categorical variables (natural ordering).
    For nominal categoricals use ``impute_polyreg`` instead.

    Uses ``statsmodels.miscmodels.ordinal_model.OrderedModel`` when
    available.  If statsmodels is not installed or the model fails to
    converge, falls back to sklearn multinomial logistic regression with
    an audible ``UserWarning``.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")

    effective_strict = _resolve_strict_mode(strict_mode)
    _warn_if_large(df, "impute_polr")
    df_norm = _normalize_missing(df)

    result = df_norm.copy()

    cat_cols = [
        col for col in df_norm.columns
        if (
            len(df_norm[col].dropna().unique()) >= 3
            and not pd.api.types.is_numeric_dtype(df_norm[col])
        )
    ]
    missing_masks = {col: df_norm[col].isna() for col in cat_cols}

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel as _OrderedModel
        _HAS_STATSMODELS = True
    except ImportError:
        _HAS_STATSMODELS = False
        warnings.warn(
            "impute_polr: statsmodels not found. Falling back to multinomial "
            "logistic regression for all columns. Install statsmodels for proper "
            "ordinal logistic regression: pip install statsmodels",
            UserWarning,
            stacklevel=2,
        )

    for _ in range(max_iter):
        for col in cat_cols:
            missing_mask = missing_masks[col]
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc, X_miss_enc = _encode_feature_pair(X_obs_raw, X_miss_raw)
            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs_enc, X_miss_enc)

            y_obs = result.loc[observed_mask, col].values
            categories = _ordinal_target_categories(df_norm[col])

            imputed_ok = False
            if _HAS_STATSMODELS:
                try:
                    category_to_code = {category: index for index, category in enumerate(categories)}
                    y_obs_codes: NDArray[np.int64] = np.asarray(
                        [category_to_code[value] for value in y_obs], dtype=np.int64
                    )
                    ord_model = _OrderedModel(y_obs_codes, X_obs_clean, distr="logit")
                    fit_result = ord_model.fit(method="bfgs", disp=False)
                    proba = fit_result.predict(X_miss_clean)
                    if hasattr(proba, "values"):
                        proba = proba.values
                    proba = np.asarray(proba)
                    best_idx = np.argmax(proba, axis=1)
                    y_pred = np.array(categories)[best_idx]
                    result.loc[missing_mask, col] = y_pred
                    imputed_ok = True
                except (ValueError, np.linalg.LinAlgError, NotFittedError) as exc:
                    if effective_strict:
                        raise ImputationError(
                            column=col, strategy="polr", original=exc
                        ) from exc
                    warnings.warn(
                        f"impute_polr: OrderedModel failed for column {col!r} "
                        f"({type(exc).__name__}: {exc}). "
                        "Falling back to multinomial logistic regression.",
                        UserWarning,
                        stacklevel=2,
                    )

            if not imputed_ok:
                model = LogisticRegression(random_state=random_state, max_iter=1000)
                try:
                    model.fit(X_obs_clean, y_obs)
                    y_pred = model.predict(X_miss_clean)
                    result.loc[missing_mask, col] = y_pred
                except (ValueError, np.linalg.LinAlgError, NotFittedError) as exc:
                    if effective_strict:
                        raise ImputationError(
                            column=col, strategy="polr", original=exc
                        ) from exc
                    fallback = result[col].mode().iloc[0]
                    warnings.warn(
                        f"impute_polr: multinomial fallback also failed for "
                        f"column {col!r} "
                        f"({type(exc).__name__}: {exc}). "
                        f"Using mode={fallback!r}. Set strict_mode=True to raise.",
                        UserWarning,
                        stacklevel=2,
                    )
                    result.loc[missing_mask, col] = fallback

    return result


def impute_rf(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    strict_mode: Optional[bool] = None,
    **rf_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Random Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_iter : int, default 1
        Number of imputation passes over all columns.
    random_state : int, default 0
        Random seed passed to the Random Forest estimators.
    strict_mode : bool or None, default None
        If True, raise :class:`~missingly.exceptions.ImputationError` on
        estimator failure instead of falling back to the column mean/mode.
        If None, uses the global :attr:`missingly.config.strict_mode` setting.
        Values must be exactly ``True`` or ``False``.
    **rf_kwargs
        Additional keyword arguments forwarded to
        :class:`~sklearn.ensemble.RandomForestRegressor` and
        :class:`~sklearn.ensemble.RandomForestClassifier`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values imputed via Random Forest.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ImputationError
        If an estimator fails and ``strict_mode=True``.
    InsufficientDataError
        If a column has no observed values.
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    >>> result = impute_rf(df, random_state=0)
    >>> bool(result["a"].isna().any())
    False
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")
    _warn_if_large(df, "impute_rf")
    df = _normalize_missing(df)
    regressor = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    classifier = RandomForestClassifier(random_state=random_state, **rf_kwargs)
    return _impute_column_by_column(
        df,
        regressor=regressor,
        classifier=classifier,
        random_state=random_state,
        strategy_name="rf",
        max_iter=max_iter,
        fill_feature_nan=False,
        strict_mode=strict_mode,
    )


def impute_gb(
    df: pd.DataFrame,
    max_iter: int = 1,
    random_state: int = 0,
    strict_mode: Optional[bool] = None,
    **gb_kwargs,
) -> pd.DataFrame:
    """Impute missing values using Gradient Boosting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    max_iter : int, default 1
        Number of imputation passes over all columns.
    random_state : int, default 0
        Random seed passed to the Gradient Boosting estimators.
    strict_mode : bool or None, default None
        If True, raise :class:`~missingly.exceptions.ImputationError` on
        estimator failure instead of falling back to the column mean/mode.
        If None, uses the global :attr:`missingly.config.strict_mode` setting.
        Values must be exactly ``True`` or ``False``.
    **gb_kwargs
        Additional keyword arguments forwarded to
        :class:`~sklearn.ensemble.GradientBoostingRegressor` and
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values imputed via Gradient Boosting.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ImputationError
        If an estimator fails and ``strict_mode=True``.
    InsufficientDataError
        If a column has no observed values.
    ConfigurationError
        If ``strict_mode`` or the selected package default is not a boolean.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    >>> result = impute_gb(df, random_state=0)
    >>> bool(result["a"].isna().any())
    False
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")
    _warn_if_large(df, "impute_gb")
    df = _normalize_missing(df)
    regressor = GradientBoostingRegressor(random_state=random_state, **gb_kwargs)
    classifier = GradientBoostingClassifier(random_state=random_state, **gb_kwargs)
    return _impute_column_by_column(
        df,
        regressor=regressor,
        classifier=classifier,
        random_state=random_state,
        strategy_name="gb",
        max_iter=max_iter,
        fill_feature_nan=True,
        strict_mode=strict_mode,
    )


# ---------------------------------------------------------------------------
# KNN Gower helper (referenced by transformer.py)
# ---------------------------------------------------------------------------

def _impute_knn_gower(
    df: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute using Gower distance for mixed numeric/categorical data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (already normalised) to impute.
    n_neighbors : int, default 5
        Number of nearest neighbours.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.
    """
    from missingly._distance import gower_distance

    result = df.copy()
    dist_matrix = gower_distance(df)

    for col in df.columns:
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue
        missing_idx = np.where(missing_mask)[0]
        observed_idx = np.where(~missing_mask)[0]
        for i in missing_idx:
            row_dists = dist_matrix[i, observed_idx]
            k = min(n_neighbors, len(observed_idx))
            nn_idx = observed_idx[np.argsort(row_dists)[:k]]
            donors = df.iloc[nn_idx][col]
            if pd.api.types.is_numeric_dtype(df[col]):
                result.iloc[i, result.columns.get_loc(col)] = donors.mean()
            else:
                result.iloc[i, result.columns.get_loc(col)] = donors.mode().iloc[0]

    return result


# ---------------------------------------------------------------------------
# FittedImputer — fit on train, transform on test (simple strategies only)
# ---------------------------------------------------------------------------

class FittedImputer:
    """Lightweight fit/transform imputer for simple strategies.

    Learns per-column fill values (mean, median, or mode) from a training
    DataFrame and applies them to any subsequent DataFrame.  This prevents
    data leakage when imputing test data inside a cross-validation loop.

    .. note::
        Only ``{mean, median, mode}`` are supported — strategies whose
        fit-state is a scalar per column.  For model-based strategies
        (``knn``, ``mice``, ``rf``, ``gb``) use
        :class:`~missingly.transformer.MissinglyImputer` instead, which
        implements the full sklearn ``BaseEstimator`` / ``TransformerMixin``
        contract including ``get_params`` / ``set_params`` / ``clone``.

    Parameters
    ----------
    strategy : {"mean", "median", "mode"}
        Fill strategy to use.

    Raises
    ------
    InvalidStrategyError
        If *strategy* is not one of ``{"mean", "median", "mode"}``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import make_imputer
    >>> train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    >>> test  = pd.DataFrame({"a": [np.nan, 5.0]})
    >>> imp = make_imputer("median").fit(train)
    >>> imp.transform(test)
         a
    0  2.0
    1  5.0
    """

    def __init__(self, strategy: str = "mean") -> None:
        if strategy not in _SIMPLE_FIT_STRATEGIES:
            raise InvalidStrategyError(
                param="strategy",
                got=strategy,
                allowed=sorted(_SIMPLE_FIT_STRATEGIES),
            )
        self.strategy = strategy
        self._fit_df: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    def __repr__(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            Shows the strategy and whether the imputer has been fitted.

        Examples
        --------
        >>> imp = FittedImputer(strategy="mean")
        >>> repr(imp)
        "FittedImputer(strategy='mean', fitted=False)"
        >>> _ = imp.fit(pd.DataFrame({"a": [1.0, 2.0]}))
        >>> repr(imp)
        "FittedImputer(strategy='mean', fitted=True)"
        """
        fitted_status = self._is_fitted
        return f"FittedImputer(strategy={self.strategy!r}, fitted={fitted_status})"

    def fit(self, df: pd.DataFrame) -> "FittedImputer":
        """Learn fill values from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.  Only non-missing values are used.

        Returns
        -------
        FittedImputer
            self (for method chaining).
        """
        validate_dataframe(df, param="df")
        self._fit_df = _normalize_missing(df).copy()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned fill values to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Data to impute.  Must have the same columns as the training
            DataFrame passed to ``fit``.

        Returns
        -------
        pd.DataFrame
            Imputed copy of *df*.

        Raises
        ------
        RuntimeError
            If ``transform`` is called before ``fit``.
        """
        if not self._is_fitted or self._fit_df is None:
            raise RuntimeError(
                "FittedImputer must be fitted before calling transform. "
                "Call .fit(train_df) first."
            )
        validate_dataframe(df, param="df")
        df = _normalize_missing(df)
        result = df.copy()

        for col in df.columns:
            if not result[col].isna().any():
                continue
            if col not in self._fit_df.columns:
                continue

            train_col = self._fit_df[col].dropna()
            if len(train_col) == 0:
                continue

            if self.strategy == "mode":
                mode_vals = train_col.mode()
                if len(mode_vals) == 0:
                    continue
                fill_val = mode_vals.iloc[0]
            elif pd.api.types.is_numeric_dtype(train_col):
                if self.strategy == "median":
                    fill_val = float(np.nanmedian(train_col.values))
                else:  # mean
                    fill_val = float(np.nanmean(train_col.values))
            else:
                mode_vals = train_col.mode()
                if len(mode_vals) == 0:
                    continue
                fill_val = mode_vals.iloc[0]

            result[col] = result[col].fillna(fill_val)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* then transform *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Data to fit and impute.

        Returns
        -------
        pd.DataFrame
            Imputed copy of *df* using fill values learned from *df* itself.
        """
        return self.fit(df).transform(df)


def make_imputer(strategy: str = "mean") -> FittedImputer:
    """Factory function that returns a :class:`FittedImputer`.

    Only simple strategies are supported: ``{mean, median, mode}``.
    For model-based strategies use
    :class:`~missingly.transformer.MissinglyImputer`.

    Parameters
    ----------
    strategy : {"mean", "median", "mode"}, default "mean"
        Fill strategy.

    Returns
    -------
    FittedImputer

    Raises
    ------
    InvalidStrategyError
        If *strategy* is not one of the supported values.

    Examples
    --------
    >>> imp = make_imputer("median")
    >>> type(imp).__name__
    'FittedImputer'
    """
    return FittedImputer(strategy=strategy)
