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
            "sound. Consider impute_rf(), impute_gb(), or impute_mice() for heavy-categorical datasets.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Python None with np.nan in object columns."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].where(df[obj_cols].notna(), other=np.nan)
    return df


def _fill_nan_with_col_means(X: np.ndarray) -> np.ndarray:
    """Replace NaN entries in a 2-D array with column means.

    Used by ``GradientBoosting`` imputers which do not natively accept
    NaN in their feature matrices.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Array that may contain NaN values.

    Returns
    -------
    np.ndarray
        Copy of *X* with NaN replaced by per-column means.  Columns
        that are entirely NaN are filled with 0.
    """
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    # For all-NaN columns nanmean returns nan — fall back to 0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X


def _split_encode(
    df: pd.DataFrame,
):
    """Ordinal-encode categorical columns; return parts needed to reconstruct."""
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

    cat_array = df_imputed[cat_cols].values
    # Round to nearest integer index before inverse transform
    cat_array = np.round(cat_array).astype(float)
    # Clip to valid range for each column
    for i, col in enumerate(cat_cols):
        n_cats = len(encoder.categories_[i])
        cat_array[:, i] = np.clip(cat_array[:, i], 0, n_cats - 1)

    decoded = encoder.inverse_transform(cat_array)
    result = df_imputed.copy()
    for i, col in enumerate(cat_cols):
        orig_dtype = cat_dtypes[col]
        if hasattr(orig_dtype, "categories"):
            result[col] = pd.Categorical(decoded[:, i], categories=orig_dtype.categories)
        else:
            result[col] = decoded[:, i]
    return result


def _impute_column_by_column(
    df: pd.DataFrame,
    regressor,
    classifier,
    random_state: int,
    max_iter: int = 1,
    fill_feature_nan: bool = False,
) -> pd.DataFrame:
    """Column-by-column imputation used by RF and GB imputers."""
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
            if observed_mask.sum() == 0:
                continue

            X_train = X_full[observed_mask]
            y_train = y_full[observed_mask]
            X_pred = X_full[missing_mask.values]

            if fill_feature_nan:
                X_train = _fill_nan_with_col_means(X_train)
                X_pred = _fill_nan_with_col_means(X_pred)

            is_cat = col in cat_cols
            if is_cat:
                est = classifier
            else:
                est = regressor

            try:
                est.fit(X_train, y_train)
                preds = est.predict(X_pred)
            except Exception:
                preds = np.nanmean(y_train) * np.ones(X_pred.shape[0])

            df_result.loc[missing_mask, col] = preds

    return _decode(df_result, cat_cols, encoder, cat_dtypes)


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
    """
    df = _normalize_missing(df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    result = df.copy()
    if len(num_cols):
        imputer = SimpleImputer(strategy="mean")
        result[num_cols] = imputer.fit_transform(df[num_cols])

    if not numeric_only and len(cat_cols):
        mode_imputer = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = mode_imputer.fit_transform(df[cat_cols])

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
    """
    df = _normalize_missing(df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    result = df.copy()
    if len(num_cols):
        imputer = SimpleImputer(strategy="median")
        result[num_cols] = imputer.fit_transform(df[num_cols])

    if not numeric_only and len(cat_cols):
        mode_imputer = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = mode_imputer.fit_transform(df[cat_cols])

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
    """
    df = _normalize_missing(df)
    imputer = SimpleImputer(strategy="most_frequent")
    result = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns,
    )
    # Restore original dtypes
    for col in df.columns:
        try:
            result[col] = result[col].astype(df[col].dtype)
        except (ValueError, TypeError):
            pass
    return result


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using k-Nearest Neighbours.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_neighbors : int, default 5
        Number of nearest neighbours to use.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values imputed via KNN.
    """
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

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    max_iter : int, default 10
        Maximum number of MICE imputation iterations per chain.
    random_state : int, default 0
        Base random seed.  When *n_imputations* > 1 each chain uses
        ``random_state + i`` for ``i in range(n_imputations)``.
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

    Notes
    -----
    **Full Multiple Imputation (MI) workflow** — to produce valid pooled
    inferences you must complete all three steps:

    1. Generate *m* imputed datasets::

           dfs = impute_mice(df, n_imputations=m)

    2. Fit your analytical model on **each** imputed dataset independently
       and collect the parameter estimates and their sampling variances.

    3. Pool the results using the MI pooling utilities in
       :mod:`missingly.mi`::

           from missingly.mi import pool_scalar_estimates
           result = pool_scalar_estimates(estimates, variances)

    **End-to-end example** (simple linear regression ``y ~ x``):

    >>> import numpy as np
    >>> import pandas as pd
    >>> from missingly import impute_mice
    >>> from missingly.mi import pool_scalar_estimates
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 10, 50)
    >>> y = 2.0 + 3.0 * x + rng.normal(0, 1, 50)
    >>> y[rng.random(50) < 0.3] = np.nan
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>>
    >>> dfs = impute_mice(df, n_imputations=5, random_state=1)
    >>> beta1_ests, beta1_vars = [], []
    >>> for d in dfs:
    ...     reg = LinearRegression().fit(d[["x"]], d["y"])
    ...     beta1_ests.append(float(reg.coef_[0]))
    ...     resid = d["y"] - reg.predict(d[["x"]])
    ...     ss_x = float(((d["x"] - d["x"].mean()) ** 2).sum())
    ...     beta1_vars.append(float(np.var(resid, ddof=2)) / ss_x)
    >>>
    >>> result = pool_scalar_estimates(beta1_ests, beta1_vars)
    >>> # Pooled beta1 should be close to 3.0
    >>> 2.0 < result["q_bar"] < 4.0
    True
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
    """Lightweight fit/transform wrapper for missingly imputation strategies.

    Parameters
    ----------
    strategy : str, default 'mean'
        Imputation strategy. One of: 'mean', 'median', 'mode', 'knn',
        'mice', 'rf', 'gb'.
    **kwargs
        Additional keyword arguments passed to the underlying imputer.

    Examples
    --------
    >>> imp = make_imputer('mean')
    >>> imp.is_fitted
    False
    >>> import pandas as pd, numpy as np
    >>> train = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
    >>> imp.fit(train)
    FittedImputer(strategy='mean', fitted=True)
    >>> imp.is_fitted
    True
    """

    def __init__(self, strategy: str = "mean", **kwargs) -> None:
        self.strategy = strategy
        self.kwargs = kwargs
        self._is_fitted = False
        self._fit_df: Optional[pd.DataFrame] = None

    @property
    def is_fitted(self) -> bool:
        """bool: True after ``fit()`` has been called, False otherwise."""
        return self._is_fitted

    def fit(self, df: pd.DataFrame) -> "FittedImputer":
        """Fit on *df* (stores statistics for transform).

        Parameters
        ----------
        df : pd.DataFrame
            Training data used to compute imputation statistics.

        Returns
        -------
        FittedImputer
            self
        """
        self._fit_df = df.copy()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute *df* using statistics learned from the training data.

        The imputer is fit on the **training** DataFrame stored during
        ``fit()``, then applied to *df*.  This prevents data leakage when
        *df* is a held-out test set.

        Parameters
        ----------
        df : pd.DataFrame
            Data to impute (may differ from the training DataFrame).

        Returns
        -------
        pd.DataFrame
            Imputed copy of *df*.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        # Compute fill values from training data, apply to df.
        # We concatenate train+test, impute together, then slice out only
        # the test rows so that fill values come from training statistics.
        train_df = self._fit_df
        n_train = len(train_df)

        combined = pd.concat([train_df, df], ignore_index=True)
        imputed_combined = _dispatch_strategy(self.strategy, combined, **self.kwargs)

        # Return only the test portion, restoring the original index
        result = imputed_combined.iloc[n_train:].copy()
        result.index = df.index
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and return its imputed copy.

        Parameters
        ----------
        df : pd.DataFrame
            Data to fit on and impute.

        Returns
        -------
        pd.DataFrame
            Imputed copy of *df*.
        """
        self.fit(df)
        return _dispatch_strategy(self.strategy, df, **self.kwargs)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"FittedImputer(strategy={self.strategy!r}, {status}=True)"


def _dispatch_strategy(strategy: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Route strategy name to the appropriate impute_* function."""
    dispatch = {
        "mean": impute_mean,
        "median": impute_median,
        "mode": impute_mode,
        "knn": impute_knn,
        "mice": impute_mice,
        "rf": impute_rf,
        "gb": impute_gb,
    }
    if strategy not in dispatch:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {list(dispatch)}"
        )
    return dispatch[strategy](df, **kwargs)


def make_imputer(strategy: str = "mean", **kwargs) -> FittedImputer:
    """Factory function for FittedImputer.

    Parameters
    ----------
    strategy : str, default 'mean'
        Imputation strategy. One of: 'mean', 'median', 'mode', 'knn',
        'mice', 'rf', 'gb'.
    **kwargs
        Additional keyword arguments passed to the underlying imputer.

    Returns
    -------
    FittedImputer
        An unfitted imputer ready for fit/transform.
    """
    return FittedImputer(strategy=strategy, **kwargs)
