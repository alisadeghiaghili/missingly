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

Dtype handling
--------------
Every imputer routes its input through one internal normalisation layer
(:mod:`missingly._dtypes`).  Numeric columns, including nullable ``Int64`` /
``Float64`` and ``float32``, are modelled as ``float64`` and restored to their
original dtype (integer columns are rounded to the nearest integer by the
model-based imputers; mean and median keep a nullable integer dtype only when
the fill value is integral).  Columns without missing values are returned
unchanged.  Boolean and categorical columns keep their dtype.  Datetime,
timedelta and mixed-type ``object`` columns are mode-filled by
``impute_mean`` / ``impute_median`` / ``impute_mode`` / ``impute_pmm`` and
:class:`FittedImputer`, and raise
:class:`~missingly.exceptions.UnsupportedDtypeError` in the model-based
imputers when they contain missing values (complete ones pass through).

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
from typing import Any, Dict, List, Optional, Tuple, Union

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

from missingly._dtypes import (
    ColumnKind,
    classify_column,
    classify_frame,
    column_mode,
    ensure_supported,
    fill_missing,
    fill_with_statistic,
    is_numeric_kind,
    normalize_frame,
    restore_dtypes,
    restore_numeric,
    to_float_array,
)
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
# Opaque dtypes the previous release imputed without error; they now only warn.
_LEGACY_ENCODED = frozenset({ColumnKind.DATETIME})
_LEGACY_CONDITIONAL = frozenset({ColumnKind.DATETIME, ColumnKind.TIMEDELTA, ColumnKind.MIXED})


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


def _require_observed(df: pd.DataFrame, columns: Optional[List] = None) -> None:
    """Raise when a column that must be imputed has no observed value at all.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to inspect (missing values already normalised).
    columns : list, optional
        Columns to check; defaults to every column.

    Raises
    ------
    InsufficientDataError
        For the first checked column whose cells are all missing.

    Examples
    --------
    >>> _require_observed(pd.DataFrame({"a": [1.0, np.nan]}))
    >>> _require_observed(pd.DataFrame({"a": [np.nan]}))
    Traceback (most recent call last):
        ...
    missingly.exceptions.InsufficientDataError: Column 'a' has 0 observed value(s), but at least 1 is required.
    """
    for column in df.columns if columns is None else columns:
        if df[column].isna().all():
            raise InsufficientDataError(column=column, n_observed=0, n_required=1)


def _impute_central(
    df: pd.DataFrame,
    strategy: str,
    numeric_only: bool,
) -> pd.DataFrame:
    """Shared implementation of :func:`impute_mean` and :func:`impute_median`.

    Parameters
    ----------
    df : pd.DataFrame
        Validated input frame.
    strategy : {"mean", "median"}
        Statistic used for numeric columns.
    numeric_only : bool
        Leave non-numeric columns untouched when ``True``.

    Returns
    -------
    pd.DataFrame
        Imputed copy.  Only columns with missing values are rewritten, so
        complete columns keep their exact dtype and values.

    Raises
    ------
    InsufficientDataError
        If an incomplete column to be imputed has no observed value.

    Examples
    --------
    >>> _impute_central(pd.DataFrame({"a": [1.0, np.nan, 3.0]}), "mean", False)["a"].tolist()
    [1.0, 2.0, 3.0]
    """
    df = _normalize_missing(df)
    kinds = classify_frame(df)
    incomplete = [column for column in df.columns if df[column].isna().any()]
    numeric = [column for column in incomplete if is_numeric_kind(kinds[column])]
    others = [] if numeric_only else [c for c in incomplete if c not in numeric]
    _require_observed(df, numeric + others)

    result = df.copy()
    for column, filled in _central_fill_numeric(df, kinds, numeric, strategy).items():
        result[column] = filled
    for column in others:
        result[column] = fill_missing(df[column], column_mode(df[column], kinds[column]))
    return result


def _central_fill_numeric(
    df: pd.DataFrame,
    kinds: Dict,
    columns: List,
    strategy: str,
) -> Dict:
    """Fill numeric columns with their mean or median, restoring each dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Frame holding the incomplete numeric *columns*.
    kinds : dict
        Column kinds from :func:`classify_frame`.
    columns : list
        Incomplete numeric columns, each with at least one observed value.
    strategy : {"mean", "median"}
        Statistic passed to :class:`~sklearn.impute.SimpleImputer`.

    Returns
    -------
    dict
        Column label to filled :class:`pandas.Series` (empty when *columns* is
        empty).

    Examples
    --------
    >>> frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    >>> _central_fill_numeric(frame, classify_frame(frame), ["a"], "mean")["a"].tolist()
    [1.0, 2.0, 3.0]
    """
    if not columns:
        return {}
    # Column-major layout keeps the per-column summation order (and hence the
    # last bit of every mean) identical to a frame-based fit.
    matrix = np.asfortranarray(
        np.column_stack([to_float_array(df[column]) for column in columns])
    )
    filled = SimpleImputer(strategy=strategy).fit_transform(matrix)
    return {
        column: restore_numeric(
            df[column],
            pd.Series(filled[:, position], index=df.index),
            kinds[column],
            round_integers=False,
        )
        for position, column in enumerate(columns)
    }


def _assign_labels(
    result: pd.DataFrame,
    mask: pd.Series,
    col: str,
    labels: np.ndarray,
) -> None:
    """Write predicted class labels into the masked cells of one column.

    Extension dtypes (``boolean``, ``Int64``, ``string``, ``category``) reject
    raw NumPy labels such as ``0.0`` / ``1.0``; the labels are therefore cast
    to the column dtype first.

    Parameters
    ----------
    result : pd.DataFrame
        Frame updated in place.
    mask : pd.Series
        Boolean mask of the cells to fill.
    col : str
        Column to write.
    labels : np.ndarray
        One label per ``True`` entry of *mask*.

    Examples
    --------
    >>> frame = pd.DataFrame({"flag": pd.array([True, None], dtype="boolean")})
    >>> _assign_labels(frame, frame["flag"].isna(), "flag", np.array([0.0]))
    >>> frame["flag"].tolist()
    [True, False]
    """
    dtype = result[col].dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        labels = pd.Series(labels).astype(dtype).array
    result.loc[mask, col] = labels


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


def _missing_to_nan(column: np.ndarray) -> np.ndarray:
    """Replace every missing scalar (``None``, ``pd.NA``, ``NaT``) in an object column by ``nan``.

    Parameters
    ----------
    column : np.ndarray
        One feature column; non-object arrays are returned unchanged.

    Returns
    -------
    np.ndarray
        Column in which ``float(value)`` is defined for every missing cell.

    Examples
    --------
    >>> _missing_to_nan(np.array([1, pd.NA], dtype=object)).tolist()
    [1, nan]
    """
    if column.dtype != object:
        return column
    missing = pd.isna(column)
    if not missing.any():
        return column
    cleaned = column.copy()
    cleaned[missing] = np.nan
    return cleaned


def _encode_features(X: np.ndarray) -> np.ndarray:
    """Encode a 2-D object/mixed array to float, replacing non-numeric with ordinal codes.

    Each column is processed independently:
    * Numeric-castable columns are kept as-is after float conversion; missing
      scalars such as ``pd.NA`` in nullable numeric columns count as missing,
      not as a reason to label-encode the column.
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
        col = _missing_to_nan(X[:, j])
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

    Examples
    --------
    >>> observed = np.array([[np.nan], [np.nan]])
    >>> _, incomplete = _fill_feature_matrix(observed, np.array([[np.nan]]))
    >>> incomplete.tolist()
    [[0.0]]
    """
    X_obs_clean = X_obs.copy()
    X_miss_clean = X_miss.copy()
    for j in range(X_obs_clean.shape[1]):
        observed_column = X_obs_clean[:, j]
        if np.isnan(observed_column).all():
            col_mean = 0.0
        else:
            col_mean = np.nanmean(observed_column)
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

    Examples
    --------
    >>> restored = _restore_categorical_dtype(
    ...     pd.Series(["low", "high"]), pd.StringDtype()
    ... )
    >>> str(restored.dtype)
    'string'
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

    Parameters
    ----------
    df : pd.DataFrame
        Data whose categorical columns are ordinal-encoded. Nullable extension
        missing values are normalized to ``np.nan`` for sklearn compatibility.

    Returns
    -------
    tuple
        ``(df_work, cat_cols, num_cols, encoder, cat_dtypes)``

    Examples
    --------
    >>> frame = pd.DataFrame({"number": [1.0, 2.0], "label": ["a", None]})
    >>> encoded, cat_cols, _, _, _ = _split_encode(frame)
    >>> cat_cols
    ['label']
    >>> encoded.shape
    (2, 2)
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
    """Inverse-transform ordinal-encoded columns back to their original dtype.

    Parameters
    ----------
    df_imputed : pd.DataFrame
        Numeric imputation result containing encoded categorical columns.
    cat_cols : list of str
        Categorical column names in the order used to fit ``encoder``.
    encoder : sklearn.preprocessing.OrdinalEncoder or None
        Fitted encoder returned by :func:`_split_encode`.
    cat_dtypes : dict of str to dtype
        Original dtypes keyed by categorical column name.

    Returns
    -------
    pd.DataFrame
        Decoded frame with categorical extension dtypes restored where possible.

    Examples
    --------
    >>> source = pd.DataFrame({"label": ["a", None]})
    >>> encoded, columns, _, encoder, dtypes = _split_encode(source)
    >>> _decode(encoded.fillna(0.0), columns, encoder, dtypes)["label"].tolist()
    ['a', 'a']
    """
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
        Input DataFrame (Python ``None`` already normalised to ``NaN``).
        Numeric columns are modelled as ``float64`` and restored to their
        original dtype afterwards; datetime, timedelta and mixed-type object
        columns are passed through (see :mod:`missingly._dtypes`).
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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type object column has missing
        values.
    """
    effective_strict = _resolve_strict_mode(strict_mode)

    normalized = normalize_frame(df, strategy_name, deprecated=_LEGACY_ENCODED)
    df_work, cat_cols, _, encoder, cat_dtypes = _split_encode(normalized.frame)
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

    decoded = _decode(df_result, cat_cols, encoder, cat_dtypes)
    return restore_dtypes(decoded, normalized)


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


def _drop_boolean_bounds(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Validate the columns named in ``bounds`` and drop boolean ones with a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Frame the bounds refer to.
    bounds : dict
        Column to ``(lower, upper)`` interval.

    Returns
    -------
    dict
        *bounds* without boolean columns.

    Raises
    ------
    ValueError
        If a bounded column is neither numeric nor boolean.

    Examples
    --------
    >>> frame = pd.DataFrame({"x": [1.0], "f": [True]})
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     _drop_boolean_bounds(frame, {"x": (0.0, 2.0), "f": (0.0, 1.0)})
    {'x': (0.0, 2.0)}
    """
    kept = {}
    for column, interval in bounds.items():
        kind = classify_column(df[column])
        if kind is ColumnKind.BOOLEAN:
            warnings.warn(
                f"impute_mice: bounds for boolean column {column!r} are ignored and will "
                "raise ValueError in a future release; bounds only apply to numeric columns.",
                FutureWarning,
                stacklevel=3,
            )
        elif not is_numeric_kind(kind):
            raise ValueError(f"bounds column {column!r} must be numeric")
        else:
            kept[column] = interval
    return kept


def _fill_categorical_gaps(
    result: pd.DataFrame,
    cat_cols: List,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Fill categorical cells that decoding left missing with the observed mode.

    Parameters
    ----------
    result : pd.DataFrame
        Decoded imputation result; modified in place and returned.
    cat_cols : list
        Categorical column labels.
    reference : pd.DataFrame
        Frame whose observed values define each column's mode.

    Returns
    -------
    pd.DataFrame
        *result* with every gap in ``cat_cols`` filled where *reference* has
        at least one observed value.

    Examples
    --------
    >>> decoded = pd.DataFrame({"g": ["a", np.nan]})
    >>> _fill_categorical_gaps(decoded, ["g"], pd.DataFrame({"g": ["b", "b"]}))["g"].tolist()
    ['a', 'b']
    """
    for col in cat_cols:
        observed = reference[col].dropna()
        if result[col].isna().any() and len(observed) > 0:
            result[col] = fill_missing(result[col], column_mode(observed))
    return result


def _mice_impute_array(
    df_work: pd.DataFrame,
    missing_mask_df: pd.DataFrame,
    estimator,
    max_iter: int,
    seed: int,
    manual_loop: bool,
    visit_sequence: Optional[List[str]],
    predictor_matrix: Optional[pd.DataFrame],
    bounds: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Run one MICE chain on the encoded, all-float working frame.

    Parameters
    ----------
    df_work : pd.DataFrame
        Numeric working frame produced by :func:`_split_encode`.
    missing_mask_df : pd.DataFrame
        Cells eligible for imputation, aligned to ``df_work``.
    estimator : sklearn estimator or None
        Caller-supplied regressor, or ``None`` for posterior-sampling
        ``BayesianRidge``.
    max_iter : int
        Number of sweeps.
    seed : int
        Chain seed.
    manual_loop : bool
        Use :func:`_run_mice_with_history` (history and execution controls)
        instead of sklearn's ``IterativeImputer``.
    visit_sequence, predictor_matrix, bounds
        Execution controls forwarded to the manual loop.

    Returns
    -------
    imputed_array : np.ndarray
        Completed values, same shape as ``df_work``.
    history : dict
        Per-iteration imputed means; empty on the sklearn path.
    """
    if estimator is None:
        base_estimator = BayesianRidge()
        use_posterior = True
    else:
        base_estimator = estimator
        use_posterior = False
    if manual_loop:
        return _run_mice_with_history(
            df_work=df_work,
            missing_mask_df=missing_mask_df,
            base_estimator=base_estimator,
            max_iter=max_iter,
            seed=seed,
            use_sample_posterior=use_posterior,
            visit_sequence=visit_sequence,
            predictor_matrix=predictor_matrix,
            bounds=bounds,
        )
    imputer = IterativeImputer(
        estimator=base_estimator,
        max_iter=max_iter,
        random_state=seed,
        imputation_order="roman",
        sample_posterior=use_posterior,
    )
    return imputer.fit_transform(df_work), {}


def _mice_conditional_model_contract(
    df: pd.DataFrame,
    estimator: object,
) -> Tuple[Dict[str, str], str, Optional[str]]:
    """Describe the conditional-model family recorded in a typed MICE result.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized input frame whose dtypes determine target families.
    estimator : object
        Caller-supplied estimator, or ``None`` for the default Bayesian ridge
        approximation.

    Returns
    -------
    methods : dict of str to str
        Per-target method identifiers suitable for :class:`ImputationPlan`.
    provenance : str
        Stable summary of the conditional-model contract.
    limitation : str or None
        Explicit limitation warning when categorical targets undergo ordinal
        encoding before either the default or caller-supplied estimator.

    Examples
    --------
    >>> methods, provenance, limitation = _mice_conditional_model_contract(
    ...     pd.DataFrame({"value": [1.0], "group": ["a"]}), None
    ... )
    >>> methods["value"]
    'bayesian_ridge_numeric'
    >>> "ordinal_encoded" in methods["group"] and limitation is not None
    True
    """
    numeric_columns = set(df.select_dtypes(include=[np.number]).columns)
    has_categorical = len(numeric_columns) != len(df.columns)
    if estimator is not None:
        methods = {
            column: (
                "custom_estimator_numeric"
                if column in numeric_columns
                else "ordinal_encoded_custom_estimator_approximation"
            )
            for column in df.columns
        }
        categorical_contract = (
            "ordinal_encoded_custom_estimator_approximation"
            if has_categorical
            else "not_applicable"
        )
        limitation = (
            "Categorical targets use ordinal encoding before the caller-supplied "
            "estimator; this is not a validated multinomial or ordinal FCS kernel."
            if has_categorical
            else None
        )
        return (
            methods,
            "numeric=caller_supplied_custom_estimator; "
            f"categorical={categorical_contract}",
            limitation,
        )

    methods = {
        column: (
            "bayesian_ridge_numeric"
            if column in numeric_columns
            else "ordinal_encoded_bayesian_ridge_approximation"
        )
        for column in df.columns
    }
    categorical_contract = (
        "ordinal_encoded_bayesian_ridge_approximation"
        if has_categorical
        else "not_applicable"
    )
    limitation = (
        "Categorical targets use ordinal encoding before the default Bayesian ridge "
        "estimator; this is not a validated multinomial or ordinal FCS kernel."
        if has_categorical
        else None
    )
    return (
        methods,
        "numeric=bayesian_ridge_posterior; "
        f"categorical={categorical_contract}",
        limitation,
    )

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
    InsufficientDataError
        If a column that would be imputed has no observed value.

    Notes
    -----
    **Dtype handling.** Numeric columns (``int*``, ``float*``, nullable
    ``Int*`` / ``Float*``) receive the mean; every other column
    (``bool``, ``boolean``, ``category``, ``string``, ``object``, datetime,
    timedelta) receives its most frequent value, so these dtypes never cause an
    error.  Only columns with missing values are rewritten; complete columns
    keep their exact dtype and values.  The mean is not rounded: a nullable
    integer column keeps its dtype when every fill value is integral and
    becomes ``Float64`` otherwise.  A ``float32`` column stays ``float32``.

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
    return _impute_central(df, "mean", numeric_only)


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
    InsufficientDataError
        If a column that would be imputed has no observed value.

    Notes
    -----
    **Dtype handling.** Numeric columns (``int*``, ``float*``, nullable
    ``Int*`` / ``Float*``) receive the median; every other column
    (``bool``, ``boolean``, ``category``, ``string``, ``object``, datetime,
    timedelta) receives its most frequent value, so these dtypes never cause an
    error.  Only columns with missing values are rewritten; complete columns
    keep their exact dtype and values.  The median is not rounded: a nullable
    integer column keeps its dtype when every fill value is integral and
    becomes ``Float64`` otherwise.  A ``float32`` column stays ``float32``.

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
    return _impute_central(df, "median", numeric_only)


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
    InsufficientDataError
        If a column requiring imputation has no observed values.

    Notes
    -----
    **Dtype handling.** Every dtype is supported, including datetime (naive
    or time-zone aware), timedelta, ``boolean``, ``string``, ``category`` and
    mixed-type ``object`` columns; the column dtype is preserved.  Ties are
    broken by the smallest value, or by first occurrence for mixed-type
    ``object`` columns, which cannot be sorted.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"city": ["Berlin", np.nan, "Berlin"]})
    >>> impute_mode(df)["city"].tolist()
    ['Berlin', 'Berlin', 'Berlin']
    """
    validate_dataframe(df, param="df")
    df_norm = _normalize_missing(df)
    result = df_norm.copy()
    for col in result.columns:
        missing_mask = result[col].isna()
        if not missing_mask.any():
            continue
        if missing_mask.all():
            raise InsufficientDataError(column=col, n_observed=0, n_required=1)
        result[col] = fill_missing(result[col], column_mode(result[col]))
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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.
    InsufficientDataError
        If a column has no observed value.

    Notes
    -----
    **Dtype handling.** Numeric columns, including nullable ``Int64`` /
    ``Float64`` and ``float32``, are modelled as ``float64`` and written back
    in their original dtype; imputed cells of an integer column are rounded
    to the nearest integer (ties to even).  Columns without missing values are
    returned exactly as given.  ``bool``, ``boolean``, ``category``,
    ``string`` and ``object`` columns are imputed as categories and keep their
    dtype.  Datetime (naive or time-zone aware), timedelta and mixed-type
    ``object`` columns are not modelled: a complete one is returned unchanged
    (after the modelled columns) and is not used as a predictor, an incomplete
    one raises :class:`~missingly.exceptions.UnsupportedDtypeError`.  An
    incomplete datetime column, which earlier releases imputed as a nominal
    category, only emits a ``FutureWarning`` for now and keeps that behaviour.

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

    normalized = normalize_frame(df, "knn", deprecated=_LEGACY_ENCODED)
    _require_observed(normalized.frame)
    if normalized.frame.shape[1] == 0:
        return df.copy()
    _warn_if_knn_heavy_categorical(normalized.frame, n_neighbors)
    df_work, cat_cols, _, encoder, cat_dtypes = _split_encode(normalized.frame)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(
        imputed_array, index=df.index, columns=df_work.columns
    )
    decoded = _decode(df_imputed, cat_cols, encoder, cat_dtypes)
    return restore_dtypes(decoded, normalized)


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
        Its schema records the eligible imputation mask, so intentionally
        retained structural missing cells remain valid in completed chains. The
        plan records each default target as numeric Bayesian ridge or an
        ordinal-encoded categorical approximation. With a caller-supplied
        estimator, the plan instead records numeric custom-estimator targets
        and categorical targets that were ordinal-encoded before that
        estimator. Categorical target handling is not a validated
        multinomial or ordinal FCS kernel in either case.
    visit_sequence : list of str, optional
        Exact ordered set of columns with original missing values to update in
        every FCS sweep. ``None`` uses the legacy left-to-right order.
    predictor_matrix : pandas.DataFrame, optional
        Square boolean or 0/1 matrix indexed and columned by ``df.columns``.
        A ``True`` / ``1`` entry at ``[target, predictor]`` enables that
        predictor in the target's conditional model. The diagonal must be zero.
    where : pandas.DataFrame, optional
        Boolean mask aligned exactly to ``df``. ``True`` permits imputation of
        an originally missing cell; ``False`` retains it as missing. It must
        not contain missing values. Over-imputation of originally observed
        cells is rejected. With
        ``return_result=True``, retained cells are recorded as structural
        missingness rather than treated as incomplete chain output.
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
        If *df* is empty, *n_imputations* < 1, or an execution-control
        argument is malformed. ``where`` masks must be aligned, two-valued,
        and may enable only originally missing cells.
    InsufficientDataError
        If an enabled target column has no observed values from which an
        imputation model can learn.
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

    Notes
    -----
    **Dtype handling.** Numeric columns, including nullable ``Int64`` /
    ``Float64`` and ``float32``, are modelled as ``float64`` and written back
    in their original dtype; imputed cells of an integer column are rounded
    to the nearest integer (ties to even).  Columns without missing values are
    returned exactly as given.  ``bool``, ``boolean``, ``category``,
    ``string`` and ``object`` columns are imputed as categories and keep their
    dtype.  Datetime (naive or time-zone aware), timedelta and mixed-type
    ``object`` columns are not modelled: a complete one is returned unchanged
    (after the modelled columns) and is not used as a predictor, an incomplete
    one raises :class:`~missingly.exceptions.UnsupportedDtypeError`.  An
    incomplete datetime column, which earlier releases imputed as a nominal
    category, only emits a ``FutureWarning`` for now and keeps that behaviour.

    ``bounds`` may only name numeric (integer or float) columns; boolean
    columns are rejected.  With ``return_result=True`` the completed frames
    follow the original column order and dtypes exactly.

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
    >>> typed.data.plan.methods["a"]
    'bayesian_ridge_numeric'
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
        if where.isna().to_numpy().any():
            raise ValueError("where must not contain missing values")
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
        bounds = _drop_boolean_bounds(df_norm, bounds)
        for column, interval in bounds.items():
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("each bounds interval must be a (lower, upper) tuple")
            lower, upper = interval
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in interval):
                raise ValueError("bounds must contain finite numeric limits")
            if lower > upper:
                raise ValueError("bounds lower limit must not exceed upper limit")

    normalized = normalize_frame(
        df_norm, "mice", targets=missing_targets, deprecated=_LEGACY_ENCODED
    )
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
        df_work, cat_cols, _, encoder, cat_dtypes = _split_encode(normalized.frame)
        capture_history = return_history or return_result
        # The manual loop records history and honours the execution controls;
        # a frame without modelled columns has nothing for sklearn to fit.
        manual_loop = (
            capture_history
            or df_work.shape[1] == 0
            or any(
                option is not None
                for option in (visit_sequence, predictor_matrix, where, bounds)
            )
        )
        imputed_array, history = _mice_impute_array(
            df_work,
            imputation_mask.loc[:, df_work.columns],
            estimator,
            max_iter,
            seed,
            manual_loop,
            visit_sequence,
            predictor_matrix,
            bounds,
        )
        df_imputed = pd.DataFrame(
            imputed_array, index=df_norm.index, columns=df_work.columns
        )
        decoded = _fill_categorical_gaps(
            _decode(df_imputed, cat_cols, encoder, cat_dtypes), cat_cols, df_norm
        )
        result = restore_dtypes(decoded, normalized)
        if return_result:
            result = result.loc[:, list(df_norm.columns)]
        if retained_missing_mask.any(axis=None):
            result = result.mask(retained_missing_mask.loc[:, result.columns])
        if capture_history:
            return result, history
        return result

    if return_result:
        results = [_single_chain(random_state + index) for index in range(n_imputations)]
        frames, histories = zip(*results)
        methods, conditional_contract, limitation = _mice_conditional_model_contract(
            df_norm,
            estimator,
        )
        schema = MissingnessSchema.from_dataframe(
            df,
            imputation_mask=imputation_mask,
        )
        provenance = {
            "engine": "impute_mice",
            "history": "per_iteration_mean",
            "conditional_model_contract": conditional_contract,
        }
        if where is not None:
            provenance["where_mask"] = "original_missing_subset"
        plan = ImputationPlan(
            methods=methods,
            n_imputations=n_imputations,
            max_iter=max_iter,
            seed_sequence=tuple(random_state + index for index in range(n_imputations)),
            provenance=provenance,
        )
        data = MIData(df.copy(deep=True), schema, plan, tuple(frames))
        contract_histories = tuple(
            {column: tuple(float(value) for value in trace) for column, trace in history.items()}
            for history in histories
        )
        warnings = [
            "Convergence has not been assessed. Run mice_convergence on the "
            "returned histories before interpreting chain stability.",
        ]
        if limitation is not None:
            warnings.append(limitation)
        return ImputationResult(
            data=data,
            histories=contract_histories,
            warnings=tuple(warnings),
            validation={
                "structural_validation_passed": True,
                "convergence_assessed": False,
                "structural_missingness_retained": bool(
                    retained_missing_mask.any(axis=None)
                ),
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
    regression imputation. When a numeric target has no numeric predictors,
    PMM uses a reproducible random hot-deck draw from its observed values.

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
    InsufficientDataError
        If a column requiring imputation has no observed donor values.

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

    **Dtype handling.** Numeric columns, including nullable ``Int64`` /
    ``Float64`` and ``float32``, are modelled as ``float64``; donor values are
    observed values, so the original dtype is restored exactly.  Every
    non-numeric column (``bool``, ``boolean``, ``category``, ``string``,
    ``object``, datetime, timedelta) is filled with its most frequent value
    and keeps its dtype.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")
    validate_positive_int(n_nearest_donors, param="n_nearest_donors")

    _warn_if_large(df, "impute_pmm")
    df_norm = _normalize_missing(df)
    _require_observed(df_norm, [c for c in df_norm.columns if df_norm[c].isna().any()])

    kinds = classify_frame(df_norm)
    num_cols = [col for col in df_norm.columns if is_numeric_kind(kinds[col])]
    if not num_cols:
        return impute_mode(df)

    rng = np.random.default_rng(random_state)
    # Numeric columns (including nullable ones) are modelled as float64 / NaN.
    work = pd.DataFrame(
        {col: to_float_array(df_norm[col]) for col in num_cols}, index=df_norm.index
    )
    missing_masks = {col: df_norm[col].isna() for col in num_cols}

    for _ in range(max_iter):
        _pmm_sweep(work, num_cols, missing_masks, rng, n_nearest_donors)
    return _pmm_finalize(df_norm, work, kinds)


def _pmm_sweep(
    work: pd.DataFrame,
    num_cols: List[str],
    missing_masks: Dict[str, pd.Series],
    rng: np.random.Generator,
    n_nearest_donors: int,
) -> None:
    """Update every incomplete numeric column once, in place.

    Parameters
    ----------
    work : pd.DataFrame
        ``float64`` working frame of the numeric columns.
    num_cols : list of str
        Numeric columns, in update order.
    missing_masks : dict of str to pd.Series
        Originally missing cells per numeric column.
    rng : np.random.Generator
        Random source.
    n_nearest_donors : int
        Size of the donor pool per missing cell.

    Examples
    --------
    >>> work = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    >>> masks = {"a": pd.Series([False, True, False, False]), "b": pd.Series([False] * 4)}
    >>> _pmm_sweep(work, ["a", "b"], masks, np.random.default_rng(0), 2)
    >>> bool(work["a"].isna().any())
    False
    """
    for col in num_cols:
        if missing_masks[col].any():
            _pmm_impute_column(work, col, missing_masks[col], num_cols, rng, n_nearest_donors)


def _pmm_finalize(
    df_norm: pd.DataFrame,
    work: pd.DataFrame,
    kinds: Dict,
) -> pd.DataFrame:
    """Write the PMM result back into the original dtypes.

    Parameters
    ----------
    df_norm : pd.DataFrame
        Input frame (missing values normalised).
    work : pd.DataFrame
        Completed ``float64`` numeric columns.
    kinds : dict
        Column kinds from :func:`classify_frame`.

    Returns
    -------
    pd.DataFrame
        Copy of *df_norm* whose incomplete numeric columns come from *work*
        (original dtype restored) and whose other incomplete columns are
        filled with their most frequent value.

    Examples
    --------
    >>> source = pd.DataFrame({"a": [1.0, np.nan], "g": ["x", None]})
    >>> done = pd.DataFrame({"a": [1.0, 1.0]})
    >>> _pmm_finalize(source, done, classify_frame(source)).values.tolist()
    [[1.0, 'x'], [1.0, 'x']]
    """
    result = df_norm.copy()
    for col in df_norm.columns:
        if not df_norm[col].isna().any():
            continue
        if col in work.columns:
            result[col] = restore_numeric(df_norm[col], work[col], kinds[col])
        else:
            result[col] = fill_missing(df_norm[col], column_mode(df_norm[col], kinds[col]))
    return result


def _pmm_impute_column(
    work: pd.DataFrame,
    col: str,
    missing_mask: pd.Series,
    num_cols: List[str],
    rng: np.random.Generator,
    n_nearest_donors: int,
) -> None:
    """Run one predictive-mean-matching update of a numeric column, in place.

    Parameters
    ----------
    work : pd.DataFrame
        ``float64`` working frame of the numeric columns; ``work[col]`` is
        updated in place at the originally missing positions.
    col : str
        Target column.
    missing_mask : pd.Series
        Boolean mask of the originally missing cells of ``col``.
    num_cols : list of str
        All numeric columns; every one except *col* is a predictor.
    rng : np.random.Generator
        Random source for the coefficient draw and the donor choice.
    n_nearest_donors : int
        Size of the donor pool per missing cell.

    Examples
    --------
    >>> work = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 10.0]})
    >>> mask = pd.Series([False, False, True, False, False])
    >>> _pmm_impute_column(work, "a", mask, ["a", "b"], np.random.default_rng(0), 2)
    >>> bool(work["a"].isna().any())
    False
    """
    feature_cols = [c for c in num_cols if c != col]
    if not feature_cols:
        observed_values = work.loc[~missing_mask, col].to_numpy()
        work.loc[missing_mask, col] = rng.choice(
            observed_values,
            size=int(missing_mask.sum()),
            replace=True,
        )
        return

    observed_mask = ~missing_mask
    X_obs = work.loc[observed_mask, feature_cols].to_numpy()
    y_obs = work.loc[observed_mask, col].to_numpy()
    X_miss = work.loc[missing_mask, feature_cols].to_numpy()

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
    work.loc[miss_indices, col] = imputed_vals


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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

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

    **Dtype handling.** Only columns that are imputed as categories are
    targets.  A datetime, timedelta or mixed-type ``object`` target with
    missing values raises :class:`~missingly.exceptions.UnsupportedDtypeError`;
    predicted labels are cast to the column dtype (``boolean``, ``Int64``,
    ``string``, ``category``).  Nullable numeric predictors are encoded by
    value, not by rank.
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
    ensure_supported(
        df_norm, "logreg", binary_cols, deprecated=_LEGACY_CONDITIONAL, deprecated_tz=False
    )
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
                _assign_labels(result, missing_mask, col, imputed)
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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

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

    **Dtype handling.** Only columns that are imputed as categories are
    targets.  A datetime, timedelta or mixed-type ``object`` target with
    missing values raises :class:`~missingly.exceptions.UnsupportedDtypeError`;
    predicted labels are cast to the column dtype (``boolean``, ``Int64``,
    ``string``, ``category``).  Nullable numeric predictors are encoded by
    value, not by rank.
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
    ensure_supported(
        df_norm, "polyreg", cat_cols, deprecated=_LEGACY_CONDITIONAL, deprecated_tz=False
    )
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
                _assign_labels(result, missing_mask, col, y_pred)
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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

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
    an audible ``UserWarning``. A one-column target has no predictors, so it
    uses its observed mode directly without importing a model backend.
    That valid baseline is retained when ``strict_mode=True`` because no
    conditional estimator has failed.

    **Dtype handling.** Only columns that are imputed as categories are
    targets.  A datetime, timedelta or mixed-type ``object`` target with
    missing values raises :class:`~missingly.exceptions.UnsupportedDtypeError`;
    predicted labels are cast to the column dtype (``boolean``, ``Int64``,
    ``string``, ``category``).  Nullable numeric predictors are encoded by
    value, not by rank.
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
    ensure_supported(
        df_norm, "polr", cat_cols, deprecated=_LEGACY_CONDITIONAL, deprecated_tz=False
    )
    missing_masks = {col: df_norm[col].isna() for col in cat_cols}

    requires_ordered_model = any(
        mask.any() and len(df_norm.columns) > 1
        for mask in missing_masks.values()
    )
    _HAS_STATSMODELS = False
    if requires_ordered_model:
        try:
            from statsmodels.miscmodels.ordinal_model import (
                OrderedModel as _OrderedModel,
            )

            _HAS_STATSMODELS = True
        except ImportError:
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
                    _assign_labels(result, missing_mask, col, y_pred)
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
                    _assign_labels(result, missing_mask, col, y_pred)
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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

    Notes
    -----
    **Dtype handling.** Numeric columns, including nullable ``Int64`` /
    ``Float64`` and ``float32``, are modelled as ``float64`` and written back
    in their original dtype; imputed cells of an integer column are rounded
    to the nearest integer (ties to even).  Columns without missing values are
    returned exactly as given.  ``bool``, ``boolean``, ``category``,
    ``string`` and ``object`` columns are imputed as categories and keep their
    dtype.  Datetime (naive or time-zone aware), timedelta and mixed-type
    ``object`` columns are not modelled: a complete one is returned unchanged
    (after the modelled columns) and is not used as a predictor, an incomplete
    one raises :class:`~missingly.exceptions.UnsupportedDtypeError`.  An
    incomplete datetime column, which earlier releases imputed as a nominal
    category, only emits a ``FutureWarning`` for now and keeps that behaviour.

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
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type ``object`` column has missing
        values.

    Notes
    -----
    **Dtype handling.** Numeric columns, including nullable ``Int64`` /
    ``Float64`` and ``float32``, are modelled as ``float64`` and written back
    in their original dtype; imputed cells of an integer column are rounded
    to the nearest integer (ties to even).  Columns without missing values are
    returned exactly as given.  ``bool``, ``boolean``, ``category``,
    ``string`` and ``object`` columns are imputed as categories and keep their
    dtype.  Datetime (naive or time-zone aware), timedelta and mixed-type
    ``object`` columns are not modelled: a complete one is returned unchanged
    (after the modelled columns) and is not used as a predictor, an incomplete
    one raises :class:`~missingly.exceptions.UnsupportedDtypeError`.  An
    incomplete datetime column, which earlier releases imputed as a nominal
    category, only emits a ``FutureWarning`` for now and keeps that behaviour.

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

    normalized = normalize_frame(df, "knn", deprecated=_LEGACY_CONDITIONAL)
    frame = normalized.frame
    _require_observed(frame)
    if frame.shape[1] == 0:
        return df.copy()
    categorical = [c for c in frame.columns if not is_numeric_kind(normalized.kinds[c])]
    dist_matrix = gower_distance(frame, cat_cols=categorical)

    result = frame.copy()
    for col in frame.columns:
        if frame[col].isna().any():
            result[col] = _gower_fill_column(
                frame[col], dist_matrix, n_neighbors, normalized.kinds[col]
            )
    return restore_dtypes(result, normalized, input_order=True)


def _gower_fill_column(
    series: pd.Series,
    dist_matrix: np.ndarray,
    n_neighbors: int,
    kind: ColumnKind,
) -> pd.Series:
    """Fill one incomplete column from its Gower-nearest observed donors.

    Parameters
    ----------
    series : pd.Series
        Modelled column (``float64`` for numeric kinds, original dtype
        otherwise) with at least one observed and one missing cell.
    dist_matrix : np.ndarray
        Pairwise Gower distances between all rows.
    n_neighbors : int
        Number of donors per missing cell.
    kind : ColumnKind
        Kind of the column.  Numeric columns receive the donor mean; boolean
        and categorical columns the donor mode.

    Returns
    -------
    pd.Series
        Filled column with the same dtype as *series*.

    Examples
    --------
    >>> column = pd.Series([1.0, np.nan, 3.0])
    >>> distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    >>> _gower_fill_column(column, distances, 2, ColumnKind.FLOAT).tolist()
    [1.0, 2.0, 3.0]
    """
    missing = series.isna().to_numpy()
    missing_idx = np.flatnonzero(missing)
    observed_idx = np.flatnonzero(~missing)
    k = min(n_neighbors, len(observed_idx))
    numeric = is_numeric_kind(kind)
    fills = []
    for i in missing_idx:
        nn_idx = observed_idx[np.argsort(dist_matrix[i, observed_idx])[:k]]
        donors = series.iloc[nn_idx]
        fills.append(donors.mean() if numeric else column_mode(donors, kind))
    filled = series.copy()
    filled.iloc[missing_idx] = fills
    return filled


# ---------------------------------------------------------------------------
# FittedImputer — fit on train, transform on test (simple strategies only)
# ---------------------------------------------------------------------------

def _learn_fill_value(strategy: str, observed: pd.Series) -> Tuple[Any, bool]:
    """Learn the scalar fill value of one training column.

    Parameters
    ----------
    strategy : {"mean", "median", "mode"}
        Fill strategy of the imputer.
    observed : pd.Series
        Non-missing training values of the column (never empty).

    Returns
    -------
    value : object
        The mean or median (a ``float``) for numeric columns under
        ``"mean"`` / ``"median"``, otherwise the mode.
    is_statistic : bool
        ``True`` when *value* is a real-valued statistic rather than an
        observed value.  Boolean, datetime, timedelta, categorical and string
        columns always use their mode, exactly as :func:`impute_mean` does.

    Examples
    --------
    >>> _learn_fill_value("mean", pd.Series([1, 2], dtype="Int64"))
    (1.5, True)
    >>> _learn_fill_value("mean", pd.Series([True, True, False]))
    (np.True_, False)
    """
    kind = classify_column(observed)
    if strategy != "mode" and is_numeric_kind(kind):
        values = to_float_array(observed)
        statistic = np.nanmedian(values) if strategy == "median" else np.nanmean(values)
        return float(statistic), True
    return column_mode(observed, kind), False


class FittedImputer:
    """Lightweight fit/transform imputer for simple strategies.

    Learns per-column fill values (mean, median, or mode) from a training
    DataFrame and applies them to any subsequent DataFrame.  This prevents
    data leakage when imputing test data inside a cross-validation loop. Input
    schemas use unique labels; transform accepts a reordering of those labels.

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

    Notes
    -----
    **Dtype handling.** The fill rule matches :func:`impute_mean` /
    :func:`impute_median`: numeric columns (including nullable ``Int64`` /
    ``Float64`` and ``float32``) use the learned mean or median, every other
    column (``bool``, ``boolean``, ``category``, ``string``, ``object``,
    datetime, timedelta) uses its learned mode.  A nullable integer column
    keeps its dtype when the fill is integral and becomes ``Float64``
    otherwise; complete columns are never rewritten.

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

        Raises
        ------
        TypeError
            If ``df`` is not a :class:`pandas.DataFrame`.
        ValueError
            If ``df`` is empty or has duplicate column labels.
        InsufficientDataError
            If a training column has no observed value from which to learn a fill.

        Examples
        --------
        >>> train = pd.DataFrame({"score": [1.0, np.nan, 3.0]})
        >>> imputer = FittedImputer("mean").fit(train)
        >>> imputer._is_fitted
        True
        """
        validate_dataframe(df, param="df")
        if not df.columns.is_unique:
            raise ValueError("fit df columns must be unique")
        df_norm = _normalize_missing(df)
        for col in df_norm.columns:
            n_observed = int(df_norm[col].notna().sum())
            if n_observed == 0:
                raise InsufficientDataError(
                    column=col,
                    n_observed=0,
                    n_required=1,
                )
        self._fit_df = df_norm.copy()
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
        TypeError
            If ``df`` is not a :class:`pandas.DataFrame`.
        ValueError
            If ``df`` is empty, has duplicate labels, or its columns do not
            match the training schema.

        Examples
        --------
        >>> train = pd.DataFrame({"score": [1.0, 3.0]})
        >>> test = pd.DataFrame({"score": [np.nan, 5.0]})
        >>> FittedImputer("mean").fit(train).transform(test)["score"].tolist()
        [2.0, 5.0]
        """
        if not self._is_fitted or self._fit_df is None:
            raise RuntimeError(
                "FittedImputer must be fitted before calling transform. "
                "Call .fit(train_df) first."
            )
        validate_dataframe(df, param="df")
        if not df.columns.is_unique:
            raise ValueError("transform df columns must be unique")
        df = _normalize_missing(df)
        training_columns = tuple(self._fit_df.columns)
        input_columns = tuple(df.columns)
        missing_columns = [col for col in training_columns if col not in input_columns]
        unexpected_columns = [
            col for col in input_columns if col not in training_columns
        ]
        if missing_columns or unexpected_columns:
            raise ValueError(
                "transform df columns must match the training columns; "
                f"missing={missing_columns}, unexpected={unexpected_columns}"
            )
        result = df.copy()

        for col in df.columns:
            if not result[col].isna().any():
                continue
            fill_value, is_statistic = _learn_fill_value(
                self.strategy, self._fit_df[col].dropna()
            )
            result[col] = (
                fill_with_statistic(result[col], fill_value)
                if is_statistic
                else fill_missing(result[col], fill_value)
            )

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

        Examples
        --------
        >>> df = pd.DataFrame({"score": [1.0, np.nan, 3.0]})
        >>> FittedImputer("median").fit_transform(df)["score"].tolist()
        [1.0, 2.0, 3.0]
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
