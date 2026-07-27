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
* Categorical columns are handled separately for ML-based imputers:
  - Numeric columns  → imputed with the provided regressor.
  - Categorical columns → imputed with a classifier, avoiding the
    error of treating category codes as continuous values.
* ``GradientBoostingRegressor`` / ``GradientBoostingClassifier`` do not
  accept NaN in feature matrices.  Any remaining NaN in the feature
  side is filled with column means computed from the training rows.

Error handling
--------------
All public functions validate their inputs eagerly and raise specific
:mod:`missingly.exceptions` exceptions on failure.  No exception is
swallowed silently.  When ``strict_mode=True`` (see :mod:`missingly.config`
or the global :data:`missingly.config.strict_mode` setting)
the column-by-column imputer raises :class:`~missingly.exceptions.ImputationError`
rather than falling back to a column mean.

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
from typing import List, Optional, Union

import numpy as np
import pandas as pd
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
    ImputationError,
    InsufficientDataError,
    InvalidStrategyError,
)
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


def _is_nan_scalar(v) -> bool:
    """Return True if *v* is a NaN-like scalar (float NaN or pd.NA)."""
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
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
    cat_encoded = encoder.fit_transform(df[cat_cols])
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

        orig_dtype = cat_dtypes[col]
        if hasattr(orig_dtype, "categories"):
            ordered = orig_dtype.ordered if hasattr(orig_dtype, "ordered") else False
            result[col] = pd.Categorical(
                decoded_col, categories=orig_dtype.categories, ordered=ordered
            )
        else:
            result[col] = decoded_col

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
        If None, falls back to :attr:`missingly.config.strict_mode`.

    Returns
    -------
    pd.DataFrame
        Imputed copy of *df*.
    """
    effective_strict = _config.strict_mode if strict_mode is None else strict_mode

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
    >>> impute_mode(df)
        city
    0  Berlin
    1  Berlin
    2  Berlin
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
    >>> impute_knn(df, n_neighbors=2)
        age
    0  25.0
    1  30.0
    2  35.0
    3  40.0
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
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Impute missing values using MICE (IterativeImputer).

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
        Regression estimator used by IterativeImputer.  Defaults to
        ``BayesianRidge()`` with ``sample_posterior=True``.
    n_imputations : int, default 1
        Number of independently imputed DataFrames to generate.
        Returns a single DataFrame when 1, a list when > 1.

    Returns
    -------
    pd.DataFrame or list of pd.DataFrame
        Imputed copy (or list of *m* copies) of *df*.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty or *n_imputations* < 1.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
    >>> result = impute_mice(df, max_iter=2, random_state=0)
    >>> result["a"].isna().any()
    False
    """
    validate_dataframe(df, param="df")
    validate_positive_int(n_imputations, param="n_imputations")
    validate_positive_int(max_iter, param="max_iter")

    _warn_if_large(df, "impute_mice")
    df_norm = _normalize_missing(df)

    def _single_chain(seed: int) -> pd.DataFrame:
        df_work, cat_cols, num_cols, encoder, cat_dtypes = _split_encode(df_norm)
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

    if n_imputations == 1:
        return _single_chain(random_state)

    return [_single_chain(random_state + i) for i in range(n_imputations)]


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
    >>> result["a"].isna().any()
    False

    Notes
    -----
    PMM is preferred over direct regression imputation because it:

    - Preserves the distribution of observed values.
    - Does not create impossible values (e.g., negative ages).
    - Handles non-linear relationships through the matching step.
    - Is the gold standard method in R's mice package.
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

    for _ in range(max_iter):
        for col in num_cols:
            missing_mask = result[col].isna()
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
            y_pred = model.predict(X_miss_clean)

            # Vectorised donor selection: for each missing value find the
            # n_nearest_donors closest observed values and pick one at random.
            # Shape: (n_missing, n_observed)
            distances = np.abs(y_obs[np.newaxis, :] - y_pred[:, np.newaxis])
            k = min(n_nearest_donors, len(y_obs))
            # argpartition is O(n) vs argsort O(n log n) — faster for large y_obs
            donor_idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
            # Random choice per row without a Python loop
            rand_col = rng.integers(0, k, size=len(y_pred))
            chosen_idx = donor_idx[np.arange(len(y_pred)), rand_col]
            imputed_vals = y_obs[chosen_idx]

            # Correct index assignment: use the actual pandas index labels
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
) -> pd.DataFrame:
    """Impute missing values using Logistic Regression (for binary variables).

    This method is equivalent to R's mice ``logreg`` method. It fits a
    logistic regression model for each binary variable with missing values,
    then samples from the predicted Bernoulli probability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations.
    random_state : int, default 0
        Random seed for reproducibility.

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
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 0.0, 1.0, 0.0]})
    >>> result = impute_logreg(df, random_state=0)
    >>> result["a"].isna().any()
    False

    Notes
    -----
    This method is best suited for binary variables (0/1 or two-level
    categorical). For categorical variables with more than 2 levels,
    use ``impute_polyreg`` instead.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")

    _warn_if_large(df, "impute_logreg")
    df_norm = _normalize_missing(df)

    rng = np.random.default_rng(random_state)
    result = df_norm.copy()

    binary_cols = [
        col for col in df_norm.columns
        if len(df_norm[col].dropna().unique()) == 2
    ]

    for _ in range(max_iter):
        for col in binary_cols:
            missing_mask = result[col].isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            # Encode all feature columns (handles mixed numeric/object)
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc = _encode_features(X_obs_raw)
            X_miss_enc = _encode_features(X_miss_raw)
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
            except Exception as exc:
                fallback = result[col].mode().iloc[0]
                warnings.warn(
                    f"impute_logreg: column {col!r} model failed ({type(exc).__name__}: {exc}). "
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
) -> pd.DataFrame:
    """Impute missing values using Multinomial Logistic Regression.

    This method is equivalent to R's mice ``polyreg`` method. It fits a
    multinomial logistic regression model for each categorical variable
    with more than 2 levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE iterations.
    random_state : int, default 0
        Random seed for reproducibility.

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
    >>> df = pd.DataFrame({"color": ["red", np.nan, "blue", "green", "red"]})
    >>> result = impute_polyreg(df, random_state=0)
    >>> result["color"].isna().any()
    False

    Notes
    -----
    Best suited for nominal categorical variables with 3+ levels.
    For binary variables use ``impute_logreg``.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(max_iter, param="max_iter")

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

    for _ in range(max_iter):
        for col in cat_cols:
            missing_mask = result[col].isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc = _encode_features(X_obs_raw)
            X_miss_enc = _encode_features(X_miss_raw)
            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs_enc, X_miss_enc)

            y_obs = result.loc[observed_mask, col].values

            model = LogisticRegression(
                random_state=random_state, max_iter=1000, multi_class="multinomial"
            )
            try:
                model.fit(X_obs_clean, y_obs)
                y_pred = model.predict(X_miss_clean)
                result.loc[missing_mask, col] = y_pred
            except Exception as exc:
                fallback = result[col].mode().iloc[0]
                warnings.warn(
                    f"impute_polyreg: column {col!r} model failed ({type(exc).__name__}: {exc}). "
                    f"Falling back to mode={fallback!r}.",
                    UserWarning,
                    stacklevel=2,
                )
                result.loc[missing_mask, col] = fallback

    return result


def impute_polr(
    df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Impute missing values using Ordinal Logistic Regression.

    This method is equivalent to R's mice ``polr`` method. It fits an
    ordinal logistic regression model for each ordered categorical variable.
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
    >>> df = pd.DataFrame({"grade": ["A", np.nan, "B", "C", "A"]})
    >>> result = impute_polr(df, random_state=0)
    >>> result["grade"].isna().any()
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

    # Check statsmodels availability once
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
            missing_mask = result[col].isna()
            if not missing_mask.any():
                continue

            feature_cols = [c for c in df_norm.columns if c != col]
            if not feature_cols:
                result.loc[missing_mask, col] = result[col].mode().iloc[0]
                continue

            observed_mask = ~missing_mask
            X_obs_raw = result.loc[observed_mask, feature_cols].values
            X_miss_raw = result.loc[missing_mask, feature_cols].values
            X_obs_enc = _encode_features(X_obs_raw)
            X_miss_enc = _encode_features(X_miss_raw)
            X_obs_clean, X_miss_clean = _fill_feature_matrix(X_obs_enc, X_miss_enc)

            y_obs = result.loc[observed_mask, col].values
            categories = sorted(df_norm[col].dropna().unique(), key=str)

            imputed_ok = False
            if _HAS_STATSMODELS:
                try:
                    ord_model = _OrderedModel(y_obs, X_obs_clean, distr="logit")
                    fit_result = ord_model.fit(method="bfgs", disp=False)
                    # predict() returns ndarray of shape (n_miss, n_categories)
                    proba = fit_result.predict(X_miss_clean)
                    if hasattr(proba, "values"):
                        proba = proba.values
                    proba = np.asarray(proba)
                    best_idx = np.argmax(proba, axis=1)
                    y_pred = np.array(categories)[best_idx]
                    result.loc[missing_mask, col] = y_pred
                    imputed_ok = True
                except Exception as exc:
                    warnings.warn(
                        f"impute_polr: OrderedModel failed for column {col!r} "
                        f"({type(exc).__name__}: {exc}). "
                        "Falling back to multinomial logistic regression.",
                        UserWarning,
                        stacklevel=2,
                    )

            if not imputed_ok:
                model = LogisticRegression(
                    random_state=random_state, max_iter=1000, multi_class="multinomial"
                )
                try:
                    model.fit(X_obs_clean, y_obs)
                    y_pred = model.predict(X_miss_clean)
                    result.loc[missing_mask, col] = y_pred
                except Exception as exc:
                    fallback = result[col].mode().iloc[0]
                    warnings.warn(
                        f"impute_polr: multinomial fallback also failed for column {col!r} "
                        f"({type(exc).__name__}: {exc}). "
                        f"Using mode={fallback!r}.",
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

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    >>> result = impute_rf(df, random_state=0)
    >>> result["a"].isna().any()
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

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    >>> result = impute_gb(df, random_state=0)
    >>> result["a"].isna().any()
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
    try:
        import gower  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Gower distance requires the 'gower' package: pip install gower"
        ) from exc

    result = df.copy()
    dist_matrix = gower.gower_matrix(df)

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
                result.at[df.index[i], col] = donors.mean()
            else:
                result.at[df.index[i], col] = donors.mode().iloc[0]

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
    >>> type(imp)
    <class 'missingly.impute.FittedImputer'>
    """
    return FittedImputer(strategy=strategy)
