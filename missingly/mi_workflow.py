"""High-level Multiple Imputation workflow helpers.

This module mirrors the ``mice::with()`` / ``mice::pool()`` pattern from R:

1. Generate *m* imputed DataFrames with :func:`~missingly.impute.impute_mice`.
2. Fit a model on each imputed dataset via :func:`mi_with` or the
   all-in-one :func:`mi_fit`.
3. Pool the scalar coefficient estimates with
   :func:`~missingly.mi.pool_scalar_estimates` — or use the convenience
   wrapper :func:`pool_linear_models` for sklearn ``LinearRegression``.

Design principles
-----------------
* ``mi_with`` is intentionally model-agnostic: ``model_fn`` can return a
  fitted sklearn estimator, a plain dict of statistics, or any object.
* ``pool_linear_models`` is a thin convenience layer; users who need to pool
  non-linear or multi-parameter models should call
  :func:`~missingly.mi.pool_scalar_estimates` or
  :func:`~missingly.mi.pool_linear_regression_results` directly.
* No new hard dependencies are introduced — sklearn is an existing soft
  dependency (required only for ``pool_linear_models``).

References
----------
van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
Section 5.1 — ``with.mids``.  https://stefvanbuuren.name/fimd/
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .impute import impute_mice
from .mi import pool_scalar_estimates


def mi_with(
    dfs: Sequence[pd.DataFrame],
    model_fn: Callable[[pd.DataFrame], Any],
) -> List[Any]:
    """Apply ``model_fn`` separately to each imputed DataFrame.

    This is a thin loop that mirrors ``mice::with()`` in R: it calls
    *model_fn* once per imputed dataset and collects the results.

    Parameters
    ----------
    dfs : sequence of pd.DataFrame
        The *m* imputed DataFrames, e.g. the output of
        ``impute_mice(df, n_imputations=m)``.
    model_fn : callable
        A function that accepts a single ``pd.DataFrame`` and returns any
        result — a fitted sklearn estimator, a dict of estimates, a
        statsmodels results object, etc.

    Returns
    -------
    list
        List of length *m* containing the return value of ``model_fn``
        called on each imputed DataFrame.

    Raises
    ------
    ValueError
        If *dfs* is empty.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from missingly import impute_mice
    >>> from missingly.mi_workflow import mi_with
    >>>
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.uniform(0, 10, 50),
    ...                    "y": 2 * rng.uniform(0, 10, 50) + rng.normal(0, 1, 50)})
    >>> df.loc[rng.choice(50, 10, replace=False), "y"] = np.nan
    >>>
    >>> dfs = impute_mice(df, n_imputations=3, random_state=0)
    >>> models = mi_with(dfs, lambda d: LinearRegression().fit(d[["x"]], d["y"]))
    >>> len(models)
    3
    >>> type(models[0]).__name__
    'LinearRegression'
    """
    dfs = list(dfs)
    if not dfs:
        raise ValueError("mi_with: dfs must contain at least one DataFrame.")
    return [model_fn(df) for df in dfs]


def mi_fit(
    df: pd.DataFrame,
    n_imputations: int,
    model_fn: Callable[[pd.DataFrame], Any],
    impute_kwargs: Optional[dict] = None,
) -> Tuple[List[pd.DataFrame], List[Any]]:
    """Convenience wrapper: impute + fit models in one call.

    Runs :func:`~missingly.impute.impute_mice` on *df* to generate
    *n_imputations* complete DataFrames, then applies *model_fn* to
    each one via :func:`mi_with`.

    Parameters
    ----------
    df : pd.DataFrame
        Incomplete DataFrame that may contain missing values.
    n_imputations : int
        Number of imputed datasets to generate (must be >= 2 for pooling).
    model_fn : callable
        Function that takes a single completed ``pd.DataFrame`` and
        returns any model result.
    impute_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        :func:`~missingly.impute.impute_mice` (e.g.
        ``{"random_state": 42, "max_iter": 5}``).

    Returns
    -------
    tuple
        ``(dfs, models)`` where:

        * ``dfs`` — list of *n_imputations* fully-imputed DataFrames.
        * ``models`` — list of *n_imputations* ``model_fn`` outputs.

    Raises
    ------
    ValueError
        If *n_imputations* < 2.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from missingly.mi_workflow import mi_fit
    >>>
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"x": rng.uniform(0, 10, 60),
    ...                    "y": 2 * rng.uniform(0, 10, 60) + rng.normal(0, 1, 60)})
    >>> df.loc[rng.choice(60, 15, replace=False), "y"] = np.nan
    >>>
    >>> dfs, models = mi_fit(
    ...     df, n_imputations=3,
    ...     model_fn=lambda d: LinearRegression().fit(d[["x"]], d["y"]),
    ...     impute_kwargs={"random_state": 1},
    ... )
    >>> len(dfs) == len(models) == 3
    True
    >>> all(d.isnull().sum().sum() == 0 for d in dfs)
    True
    """
    if n_imputations < 2:
        raise ValueError(
            f"mi_fit: n_imputations must be >= 2 for pooling; got {n_imputations}."
        )
    kwargs = dict(impute_kwargs or {})
    dfs = impute_mice(df, n_imputations=n_imputations, **kwargs)
    models = mi_with(dfs, model_fn)
    return dfs, models


def pool_linear_models(
    models: Sequence[Any],
    dfs: Sequence[pd.DataFrame],
    feature: str,
    target: str,
) -> Dict[str, float]:
    """Pool slope estimates from *m* fitted sklearn LinearRegression models.

    Extracts the slope coefficient for *feature* → *target* from each
    model, estimates its within-imputation variance using the standard
    OLS formula (``sigma² / SS_x``), and combines everything with
    :func:`~missingly.mi.pool_scalar_estimates` (Rubin's Rules).

    Parameters
    ----------
    models : sequence of fitted ``LinearRegression``
        The *m* fitted models, e.g. the second element of
        :func:`mi_fit` output or the result of :func:`mi_with`.
    dfs : sequence of pd.DataFrame
        The *m* imputed DataFrames used to fit *models* (needed to
        compute ``SS_x`` and residual variance for each fit).
    feature : str
        Name of the single predictor column whose slope is pooled.
    target : str
        Name of the response column.

    Returns
    -------
    dict
        Output of :func:`~missingly.mi.pool_scalar_estimates` for the
        slope of *feature*.  Keys: ``"q_bar"``, ``"u_bar"``, ``"b"``,
        ``"t"``, ``"df"``, ``"r"``, ``"lambda"``, ``"fmi"``.

    Raises
    ------
    ValueError
        If *models* and *dfs* have different lengths, or fewer than 2
        imputations are provided.
    KeyError
        If *feature* or *target* is not found in a DataFrame.

    Notes
    -----
    The within-imputation variance of the OLS slope is estimated as:

    .. math::

        \\widehat{\\text{Var}}(\\hat{\\beta}_1) =
            \\frac{\\hat{\\sigma}^2}{SS_x},
            \\quad \\hat{\\sigma}^2 = \\frac{\\sum_i \\hat{e}_i^2}{n - 2}

    This is the same formula used in the integration tests in
    ``tests/test_mi_pooling.py``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from missingly.mi_workflow import mi_fit, pool_linear_models
    >>>
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({"x": rng.uniform(0, 10, 100),
    ...                    "y": 2 * rng.uniform(0, 10, 100) + rng.normal(0, 1, 100)})
    >>> df.loc[rng.choice(100, 30, replace=False), "y"] = np.nan
    >>>
    >>> dfs, models = mi_fit(
    ...     df, n_imputations=5,
    ...     model_fn=lambda d: LinearRegression().fit(d[["x"]], d["y"]),
    ...     impute_kwargs={"random_state": 42},
    ... )
    >>> pooled = pool_linear_models(models, dfs, feature="x", target="y")
    >>> 1.7 < pooled["q_bar"] < 2.3
    True
    """
    models = list(models)
    dfs = list(dfs)

    if len(models) != len(dfs):
        raise ValueError(
            f"pool_linear_models: models (len={len(models)}) and "
            f"dfs (len={len(dfs)}) must have the same length."
        )
    if len(models) < 2:
        raise ValueError(
            "pool_linear_models: at least 2 models required for pooling; "
            f"got {len(models)}."
        )

    slopes: List[float] = []
    variances: List[float] = []

    for model, df in zip(models, dfs):
        slope = float(model.coef_[0])

        x_vals = df[feature].values.astype(float)
        y_vals = df[target].values.astype(float)
        prediction_input = (
            df[[feature]]
            if hasattr(model, "feature_names_in_")
            else x_vals.reshape(-1, 1)
        )
        y_pred = model.predict(prediction_input).ravel()

        n = len(y_vals)
        resid = y_vals - y_pred
        sigma2 = float(np.sum(resid ** 2) / max(n - 2, 1))
        ss_x = float(np.sum((x_vals - x_vals.mean()) ** 2))
        var_slope = sigma2 / ss_x if ss_x > 1e-12 else 1e-6

        slopes.append(slope)
        variances.append(max(var_slope, 1e-10))

    return pool_scalar_estimates(slopes, variances)
