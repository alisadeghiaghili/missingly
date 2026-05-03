"""Imputation comparison utilities.

This module provides two complementary benchmarking functions:

:func:`compare_imputations`
    **Quick exploration tool.**  Requires a *complete* DataFrame, masks a
    fraction of values, imputes, and measures reconstruction RMSE /
    accuracy.  Fast and deterministic, but answers the wrong question for
    production use: it measures how well an imputer *reconstructs masked
    values* rather than how well the downstream model *generalises*.

:func:`cv_compare_imputations`
    **Production-quality benchmark.**  Accepts a DataFrame with *real*
    missing values and a supervised target vector.  For each CV fold it
    fits the imputer on the training split (no leakage), transforms both
    splits, and scores the downstream estimator.  Returns per-imputer
    mean and std of the cross-validated score so you can select the
    imputer that yields the best model — not the best RMSE on masked
    cells.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import is_classifier

from . import impute as _impute_module
from .transformer import MissinglyImputer


def compare_imputations(
    df: pd.DataFrame,
    methods: Optional[List] = None,
    mask_frac: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare imputation methods by masking a complete DataFrame and measuring reconstruction.

    Artificially masks a fraction of values in each column, applies each
    imputation method, and evaluates reconstruction accuracy.

    .. note::
        This function answers the question *"which imputer best reconstructs
        masked values?"*, **not** *"which imputer gives the best downstream
        model?"*.  For the latter use :func:`cv_compare_imputations`.

    Scoring:

    * **Numeric columns** → RMSE (lower is better).
    * **Categorical columns** → accuracy (higher is better).
    * **Mixed** → normalised composite ``Score`` (lower is better).

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** DataFrame (no missing values) with at least one
        column.  Mixed numeric/categorical dtypes are supported.
    methods : list of callables, optional
        Imputation functions to compare.  Each must accept a DataFrame and
        return a fully-imputed DataFrame.  Defaults to all seven built-in
        methods: mean, median, mode, knn, mice, rf, gb.
    mask_frac : float, optional
        Fraction of values to mask per column (default 0.20).  Must be
        in (0, 1).
    random_state : int, optional
        Random seed for reproducible masking.  Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by method name, sorted ascending by ``Score``
        (or ``RMSE`` / ``Accuracy`` when only one column type exists).

    Raises
    ------
    ValueError
        If *df* has missing values (a complete DataFrame is required).
    ValueError
        If *mask_frac* is not in (0, 1).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'age': [25, 30, 35, 40], 'city': ['A','B','A','B']})
    >>> compare_imputations(df)
    """
    if df.isnull().any().any():
        raise ValueError(
            "compare_imputations requires a complete DataFrame (no missing values). "
            "Call df.dropna() or use cv_compare_imputations() for data with real "
            "missing values."
        )
    if not (0.0 < mask_frac < 1.0):
        raise ValueError(f"mask_frac must be in (0, 1); got {mask_frac!r}")

    if methods is None:
        methods = [
            _impute_module.impute_mean,
            _impute_module.impute_median,
            _impute_module.impute_mode,
            _impute_module.impute_knn,
            _impute_module.impute_mice,
            _impute_module.impute_rf,
            _impute_module.impute_gb,
        ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols and not cat_cols:
        raise ValueError("DataFrame has no columns to evaluate.")

    rng = np.random.default_rng(seed=random_state)

    masks: dict = {}
    for col in numeric_cols + cat_cols:
        n_mask = max(1, int(len(df) * mask_frac))
        idx = rng.choice(df.index, size=n_mask, replace=False)
        masks[col] = idx

    df_missing = df.copy()
    for col, idx in masks.items():
        df_missing.loc[idx, col] = np.nan

    results = {}
    for method in methods:
        df_imputed = method(df_missing)
        row: dict = {}

        if numeric_cols:
            rmse_vals = []
            for col in numeric_cols:
                idx = masks[col]
                true_vals = df.loc[idx, col].values.astype(float)
                pred_vals = df_imputed.loc[idx, col].values.astype(float)
                rmse_vals.append(
                    np.sqrt(mean_squared_error(true_vals, pred_vals))
                )
            row["RMSE"] = float(np.mean(rmse_vals))

        if cat_cols:
            acc_vals = []
            for col in cat_cols:
                idx = masks[col]
                true_vals = df.loc[idx, col].values
                pred_vals = df_imputed.loc[idx, col].values
                acc_vals.append(accuracy_score(true_vals, pred_vals))
            row["Accuracy"] = float(np.mean(acc_vals))

        results[method.__name__] = row

    result_df = pd.DataFrame.from_dict(results, orient="index")

    if numeric_cols and cat_cols:
        rmse_min, rmse_max = result_df["RMSE"].min(), result_df["RMSE"].max()
        if rmse_max > rmse_min:
            norm_rmse = (result_df["RMSE"] - rmse_min) / (rmse_max - rmse_min)
        else:
            norm_rmse = pd.Series(0.0, index=result_df.index)
        result_df["Score"] = (norm_rmse + (1.0 - result_df["Accuracy"])) / 2.0
        return result_df.sort_values(by="Score")
    elif numeric_cols:
        return result_df.sort_values(by="RMSE")
    else:
        return result_df.sort_values(by="Accuracy", ascending=False)


def cv_compare_imputations(
    X: pd.DataFrame,
    y,
    estimator,
    strategies: Optional[List[str]] = None,
    n_splits: int = 5,
    scoring=None,
    random_state: int = 42,
    imputer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Compare imputation strategies via cross-validated downstream model score.

    This is the **correct** way to choose an imputer for production.  For
    each fold of a K-Fold CV:

    1. Fit the imputer on *X_train* only (no leakage).
    2. Transform *X_train* and *X_test* using those fitted parameters.
    3. Fit the downstream *estimator* on imputed *X_train*.
    4. Score on imputed *X_test*.

    The final result is the mean and standard deviation of the CV scores
    across all folds, giving a direct comparison of how each imputation
    strategy affects model generalisation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, may contain real missing values.  The imputer is
        fit on each training fold and applied to the test fold.
    y : array-like
        Target vector.  Must have the same length as *X*.
    estimator : sklearn estimator
        Any fitted sklearn estimator with a ``predict`` (and optionally
        ``predict_proba``) method.  A fresh clone is used for each fold.
    strategies : list of str, optional
        List of imputation strategy names to compare.  Each must be one
        of: ``"mean"``, ``"median"``, ``"mode"``, ``"knn"``, ``"mice"``,
        ``"rf"``, ``"gb"``.
        Defaults to ``["mean", "median", "mode", "knn", "mice"]``.
    n_splits : int, optional
        Number of CV folds.  Default 5.
    scoring : callable, optional
        A function ``scoring(estimator, X, y) -> float``.  Defaults to
        ``estimator.score`` (accuracy for classifiers, R² for regressors).
    random_state : int, optional
        Random seed for KFold shuffling.  Default 42.
    imputer_kwargs : dict, optional
        Dict mapping strategy name to keyword arguments forwarded to
        :class:`~missingly.transformer.MissinglyImputer`.  E.g.
        ``{"knn": {"n_neighbors": 3}}``.

    Returns
    -------
    pd.DataFrame
        Indexed by strategy name, with columns:

        * ``mean_score`` — mean CV score across folds (higher is better
          for classifiers / R²; depends on *scoring*).
        * ``std_score``  — standard deviation of CV scores.

        Sorted descending by ``mean_score``.

    Raises
    ------
    ValueError
        If *strategies* contains an unrecognised strategy name.
    ValueError
        If *X* and *y* have different lengths.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from missingly.compare import cv_compare_imputations
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame({'a': rng.normal(size=100), 'b': rng.normal(size=100)})
    >>> X.loc[rng.choice(100, 20, replace=False), 'a'] = np.nan
    >>> y = rng.integers(0, 2, size=100)
    >>>
    >>> results = cv_compare_imputations(
    ...     X, y,
    ...     estimator=LogisticRegression(),
    ...     strategies=['mean', 'knn', 'mice'],
    ...     n_splits=3,
    ... )
    >>> print(results)
    """
    _VALID = frozenset({"mean", "median", "mode", "knn", "mice", "rf", "gb"})

    if strategies is None:
        strategies = ["mean", "median", "mode", "knn", "mice"]

    unknown = set(strategies) - _VALID
    if unknown:
        raise ValueError(
            f"Unknown strategies: {sorted(unknown)}. "
            f"Valid options: {sorted(_VALID)}."
        )

    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length; got {len(X)} and {len(y)}."
        )

    imputer_kwargs = imputer_kwargs or {}

    # Use stratified splits for classifiers
    try:
        _is_clf = is_classifier(estimator)
    except Exception:
        _is_clf = False

    if _is_clf:
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = kf.split(X, y)
    else:
        kf = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = kf.split(X)

    # Materialise all fold indices so we can reuse them per strategy
    folds = list(split_iter)
    X_arr = X.reset_index(drop=True)
    y_arr = y

    results = {}
    for strategy in strategies:
        kwargs = imputer_kwargs.get(strategy, {})
        fold_scores = []

        for train_idx, test_idx in folds:
            X_tr = X_arr.iloc[train_idx].copy()
            X_te = X_arr.iloc[test_idx].copy()
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]

            # Fit imputer on training fold only — no leakage
            imp = MissinglyImputer(strategy=strategy, **kwargs)
            imp.fit(X_tr)
            X_tr_imp = imp.transform(X_tr)
            X_te_imp = imp.transform(X_te)

            # Clone estimator to avoid state bleed across folds
            from sklearn.base import clone
            est = clone(estimator)
            est.fit(X_tr_imp, y_tr)

            if scoring is not None:
                fold_scores.append(float(scoring(est, X_te_imp, y_te)))
            else:
                fold_scores.append(float(est.score(X_te_imp, y_te)))

        results[strategy] = {
            "mean_score": float(np.mean(fold_scores)),
            "std_score":  float(np.std(fold_scores)),
        }

    result_df = pd.DataFrame.from_dict(results, orient="index")
    return result_df.sort_values(by="mean_score", ascending=False)
