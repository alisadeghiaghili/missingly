# This module incorporates code adapted from the XeroGraph library by Julhash Kazi
# Original source: https://github.com/kazilab/XeroGraph
# Licensed under the Apache License, Version 2.0.
# Copyright 2024 Julhash Kazi

"""Statistical diagnostics and summary statistics for missing data.

This module is the canonical home for **all** missingness analysis that
does not involve imputation or visualisation.  It merges what was
previously split across ``stats.py`` and ``summary.py``.

Public API — Summary
--------------------
bind_shadow
    Attach a boolean shadow matrix to a DataFrame.
n_miss / n_complete
    Dataset-level missing / present counts.
pct_miss / pct_complete
    Dataset-level missing / present percentages.
miss_var_summary
    Per-column missingness table.
miss_case_summary
    Per-row missingness table.
summarise
    One-row dataset-level summary DataFrame.
miss_scan_count
    Count specific sentinel values per column.
miss_summary
    Combined dict of all three summary tables.

Public API — Statistical Tests
-------------------------------
mcar_test
    Little's MCAR test (chi-square + p-value).
missingness_association_test
    Exploratory observed-data association screen; does not identify MAR or MNAR.
mar_mnar_test
    Deprecated compatibility wrapper for ``missingness_association_test``.
diagnose_missing
    MCAR evidence summary and assumption-aware planning guidance.

Public API — MICE Convergence Diagnostics
------------------------------------------
mice_convergence
    Assess convergence of multiple MICE chains using iteration history.
plot_mice_convergence
    Plot per-chain trace plots with Gelman-Rubin R-hat annotation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, solve
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from missingly._validation import validate_dataframe


# ===========================================================================
# Section 1 — Summary statistics
# ===========================================================================

def bind_shadow(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Bind a boolean shadow matrix to a DataFrame.

    The shadow matrix has one column per original column named
    ``<col>_NA``; ``True`` means the value is missing.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
        Sentinel values treated as missing in addition to NaN.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_rows, 2 * n_cols)``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'A': [1.0, np.nan], 'B': [np.nan, 2.0]})
    >>> bind_shadow(df)
         A    B   A_NA   B_NA
    0  1.0  NaN  False   True
    1  NaN  2.0   True  False
    """
    validate_dataframe(df, allow_empty=True)
    shadow = df.isnull()
    if missing_values is not None:
        shadow = shadow | df.isin(missing_values)
    shadow.columns = [f"{col}_NA" for col in df.columns]
    return pd.concat([df, shadow], axis=1)


def n_miss(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> int:
    """Count total missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
        Sentinel values treated as missing in addition to NaN.

    Returns
    -------
    int
    """
    validate_dataframe(df, allow_empty=True)
    if missing_values is None:
        return int(df.isnull().sum().sum())
    return int((df.isnull() | df.isin(missing_values)).sum().sum())


def n_complete(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> int:
    """Count total non-missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    int
    """
    validate_dataframe(df, allow_empty=True)
    return df.size - n_miss(df, missing_values)


def pct_miss(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> float:
    """Percentage of missing values (0–100).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    float
    """
    validate_dataframe(df, allow_empty=True)
    return (n_miss(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def pct_complete(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> float:
    """Percentage of non-missing values (0–100).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    float
    """
    validate_dataframe(df, allow_empty=True)
    return (n_complete(df, missing_values) / df.size) * 100 if df.size > 0 else 0.0


def miss_var_summary(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Summarise missingness by variable (column).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``variable``, ``n_miss``, ``pct_miss``.
    """
    validate_dataframe(df, allow_empty=True)
    if missing_values is None:
        miss_counts = df.isnull().sum()
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum()
    miss_pct = (miss_counts / len(df)) * 100 if len(df) > 0 else 0
    return pd.DataFrame({
        "variable": df.columns,
        "n_miss": miss_counts.values,
        "pct_miss": miss_pct.values,
    }).reset_index(drop=True)


def miss_case_summary(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Summarise missingness by case (row).

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``case``, ``n_miss``, ``pct_miss``.
    """
    validate_dataframe(df, allow_empty=True)
    if missing_values is None:
        miss_counts = df.isnull().sum(axis=1)
    else:
        miss_counts = (df.isnull() | df.isin(missing_values)).sum(axis=1)
    miss_pct = (miss_counts / df.shape[1]) * 100 if df.shape[1] > 0 else 0
    return pd.DataFrame({
        "case": df.index,
        "n_miss": miss_counts.values,
        "pct_miss": miss_pct.values,
    }).reset_index(drop=True)


def summarise(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """High-level one-row summary of missing data for the whole DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        One row with columns:
        ``n_rows``, ``n_cols``, ``n_cells``,
        ``n_miss``, ``n_complete``, ``pct_miss``, ``pct_complete``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, 2.0, 3.0]})
    >>> summarise(df)
       n_rows  n_cols  n_cells  n_miss  n_complete   pct_miss  pct_complete
    0       3       2        6       2           4  33.333333     66.666667
    """
    validate_dataframe(df, allow_empty=True)
    nm = n_miss(df, missing_values)
    nc = n_complete(df, missing_values)
    total = df.size
    return pd.DataFrame([{
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_cells": total,
        "n_miss": nm,
        "n_complete": nc,
        "pct_miss": (nm / total * 100) if total > 0 else 0.0,
        "pct_complete": (nc / total * 100) if total > 0 else 0.0,
    }])


def miss_scan_count(
    df: pd.DataFrame,
    search: Union[List[Any], Any],
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Count occurrences of specific sentinel values per variable.

    Parameters
    ----------
    df : pd.DataFrame
    search : scalar or list
        Value(s) to search for (e.g. ``[-99, -98, "N/A"]``).
    missing_values : list, optional

    Returns
    -------
    pd.DataFrame
        Long-format with columns ``variable``, ``value``, ``n``.
        Only rows where ``n > 0`` are returned.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [-99, 1, -99], 'b': [2, -99, 3]})
    >>> miss_scan_count(df, search=[-99])
      variable  value  n
    0        a    -99  2
    1        b    -99  1
    """
    validate_dataframe(df, allow_empty=True)
    if not isinstance(search, list):
        search = [search]

    rows = []
    for col in df.columns:
        series = df[col]
        for val in search:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                count = int(series.isna().sum())
            else:
                try:
                    count = int((series == val).sum())
                except TypeError:
                    count = 0
            if count > 0:
                rows.append({"variable": col, "value": val, "n": count})

    if rows:
        return pd.DataFrame(rows).reset_index(drop=True)
    return pd.DataFrame(columns=["variable", "value", "n"])


def miss_summary(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
    digits: int = 2,
) -> dict:
    """Combined missing-data summary: dataset, variable, and case views.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
    digits : int, default 2
        Decimal places for ``pct_miss`` / ``pct_complete`` columns.

    Returns
    -------
    dict
        Keys: ``"summary"``, ``"by_variable"``, ``"by_case"``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': [1.0, np.nan], 'b': [np.nan, 2.0]})
    >>> list(miss_summary(df).keys())
    ['summary', 'by_variable', 'by_case']
    """
    validate_dataframe(df, allow_empty=True)

    def _round(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in ("pct_miss", "pct_complete"):
            if col in out.columns:
                out[col] = out[col].round(digits)
        return out

    return {
        "summary": _round(summarise(df, missing_values)),
        "by_variable": _round(miss_var_summary(df, missing_values)),
        "by_case": _round(miss_case_summary(df, missing_values)),
    }


# ===========================================================================
# Section 2 — Statistical tests (from stats.py)
# ===========================================================================

def _em_mle_estimation(data, max_iter=500, tol=1e-7, ridge=1e-8):
    """Estimate multivariate-normal moments with observed-data EM.

    The M-step includes conditional covariance for the missing components. This
    is required for the maximum-likelihood moments used by Little's statistic;
    mean-imputation followed by ``numpy.cov`` is not an EM M-step.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Raw numeric array; NaN marks missing values.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Maximum absolute parameter change required for convergence.
    ridge : float
        Non-negative diagonal stabilizer for conditional covariance solves.

    Returns
    -------
    mu_hat : np.ndarray, shape (d,)
    sigma_hat : np.ndarray, shape (d, d)
    iterations : int
        Number of EM iterations performed.
    """
    X = np.asarray(data, dtype=float)
    n, d = X.shape
    col_means = np.nanmean(X, axis=0)
    if not np.isfinite(col_means).all():
        raise ValueError("mcar_test cannot estimate a column that is entirely missing.")

    completed = np.where(np.isnan(X), col_means, X)
    mu_hat = completed.mean(axis=0)
    sigma_hat = (completed - mu_hat).T @ (completed - mu_hat) / n
    sigma_hat = (sigma_hat + sigma_hat.T) / 2.0

    for iteration in range(1, max_iter + 1):
        expected_sum = np.zeros(d, dtype=float)
        expected_cross = np.zeros((d, d), dtype=float)
        for row in X:
            missing = np.isnan(row)
            observed = ~missing
            expected = mu_hat.copy()
            conditional_covariance = None
            if missing.any() and observed.any():
                sigma_oo = sigma_hat[np.ix_(observed, observed)]
                if ridge:
                    sigma_oo = sigma_oo + np.eye(observed.sum()) * ridge
                sigma_mo = sigma_hat[np.ix_(missing, observed)]
                sigma_om = sigma_hat[np.ix_(observed, missing)]
                try:
                    solved_residual = solve(sigma_oo, row[observed] - mu_hat[observed])
                    solved_covariance = solve(sigma_oo, sigma_om)
                except LinAlgError as error:
                    raise ValueError(
                        "mcar_test could not invert an observed covariance block; "
                        "remove collinear columns or use a positive ridge."
                    ) from error
                expected[missing] = mu_hat[missing] + sigma_mo @ solved_residual
                conditional_covariance = (
                    sigma_hat[np.ix_(missing, missing)] - sigma_mo @ solved_covariance
                )
            elif missing.any():
                conditional_covariance = sigma_hat[np.ix_(missing, missing)]

            expected[observed] = row[observed]
            expected_sum += expected
            expected_cross += np.outer(expected, expected)
            if conditional_covariance is not None:
                expected_cross[np.ix_(missing, missing)] += conditional_covariance

        new_mu = expected_sum / n
        new_sigma = expected_cross / n - np.outer(new_mu, new_mu)
        new_sigma = (new_sigma + new_sigma.T) / 2.0
        change = max(
            float(np.max(np.abs(new_mu - mu_hat))),
            float(np.max(np.abs(new_sigma - sigma_hat))),
        )
        mu_hat, sigma_hat = new_mu, new_sigma
        if change <= tol:
            return mu_hat, sigma_hat, iteration

    raise RuntimeError(
        "mcar_test EM algorithm did not converge within "
        f"max_iter={max_iter}; increase max_iter or inspect the data."
    )


def mcar_test(
    X: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-7,
    ridge: float = 1e-8,
    missing_values: Optional[List] = None,
) -> Dict:
    """Perform Little's MCAR test.

    Tests the null hypothesis that data are Missing Completely At Random
    (MCAR) under a multivariate-normal working model. A small p-value gives
    evidence *against* MCAR; a large p-value does not prove MCAR.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with missing values. All columns must be numeric, finite when
        observed, and contain at least one observed value.
    max_iter : int, default 500
        Maximum EM iterations.
    tol : float, default 1e-7
        Maximum absolute EM parameter change required for convergence.
    ridge : float, default 1e-8
        Non-negative diagonal stabilizer used only for conditional covariance
        solves. A positive value changes the numerical approximation.
    missing_values : list, optional
        Extra sentinel values treated as missing.

    Returns
    -------
    dict
        Keys: ``chi_square``, ``df``, ``p_value``, ``missing_patterns``,
        ``amount_missing``, and ``em_iterations``.

    Raises
    ------
    TypeError
        If *X* is not a :class:`pandas.DataFrame`.
    ValueError
        If data are invalid, non-numeric, non-finite when observed, contain an
        all-missing column, or do not have enough informative rows.
    RuntimeError
        If observed-data EM does not converge within ``max_iter``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [1.0, np.nan, 3.0, 4.0]})
    >>> "p_value" in mcar_test(frame)
    True

    Notes
    -----
    This is not a general test for MAR versus MNAR, and non-rejection is not a
    basis for claiming MCAR or selecting a single imputation method.

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. *Journal of the American Statistical
    Association*, 83(404), 1198-1202. https://doi.org/10.1080/01621459.1988.10478722
    """
    validate_dataframe(X, param="X")
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    if not X.isnull().any(axis=None):
        raise ValueError(
            "mcar_test requires at least one missing value in the DataFrame. "
            "All values are present — MCAR test is not applicable."
        )

    if X.shape[1] < 2:
        raise ValueError("mcar_test requires at least two numeric columns.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("mcar_test requires numeric columns; encode categoricals explicitly.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    data_np = X.to_numpy(dtype=float)
    if np.isinf(data_np).any():
        raise ValueError("mcar_test does not accept infinite observed values.")
    n, d = data_np.shape
    if np.isnan(data_np).all(axis=0).any():
        raise ValueError("mcar_test cannot estimate a column that is entirely missing.")
    if (~np.isnan(data_np).all(axis=1)).sum() <= d:
        raise ValueError(
            "mcar_test requires more informative rows than numeric columns."
        )

    r = np.isnan(data_np).astype(int)
    nmis = r.sum(axis=0)
    powers_of_2 = 2 ** np.arange(d)
    mdp = r.dot(powers_of_2) + 1

    df_wp = pd.DataFrame(data_np.copy(), columns=X.columns)
    df_wp["MisPat"] = mdp

    unique_patterns = np.sort(df_wp["MisPat"].unique())
    n_mis_pat = len(unique_patterns)

    mu_hat, Sigma_hat, em_iterations = _em_mle_estimation(
        data_np, max_iter=max_iter, tol=tol, ridge=ridge
    )

    pattern_map = {pat: i + 1 for i, pat in enumerate(unique_patterns)}
    df_wp["MisPat"] = df_wp["MisPat"].map(pattern_map)

    datasets = {
        pat_id: df_wp[df_wp["MisPat"] == pat_id].iloc[:, :d]
        for pat_id in range(1, n_mis_pat + 1)
    }

    df_val = sum(
        (datasets[p].isna().sum(axis=0) == 0).sum()
        for p in range(1, n_mis_pat + 1)
    ) - d

    d2 = 0.0
    for pat_id in range(1, n_mis_pat + 1):
        subset = datasets[pat_id]
        n_pat = subset.shape[0]
        obs_cols = np.where(subset.isna().sum(axis=0) == 0)[0]
        if len(obs_cols) == 0:
            continue
        mean_diff = subset.mean(axis=0).values[obs_cols] - mu_hat[obs_cols]
        Sigma_obs = Sigma_hat[np.ix_(obs_cols, obs_cols)]
        if ridge:
            Sigma_obs = Sigma_obs + np.eye(len(obs_cols)) * ridge
        try:
            d2 += n_pat * (mean_diff @ solve(Sigma_obs, mean_diff))
        except LinAlgError as error:
            raise ValueError(
                "mcar_test could not invert a pattern covariance block; "
                "remove collinear columns or use a positive ridge."
            ) from error

    p_val = np.nan if df_val <= 0 else float(1 - chi2.cdf(d2, df_val))
    amount_missing = pd.DataFrame(
        [nmis, nmis / n],
        index=["Number Missing", "Percent Missing"],
        columns=X.columns,
    )
    return {
        "chi_square": d2,
        "df": df_val,
        "p_value": p_val,
        "missing_patterns": n_mis_pat,
        "amount_missing": amount_missing,
        "em_iterations": em_iterations,
    }


def _logistic_log_likelihood(model, X, y):
    """Log-likelihood of a fitted LogisticRegression."""
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def missingness_association_test(
    X: pd.DataFrame,
    outcome,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Screen observedness associations with an observed auxiliary outcome.

    For each feature with both observed and missing values, this diagnostic compares
    two logistic models for its observedness indicator: one using the other columns in
    ``X`` and one additionally using ``outcome``. It tests whether that *observed*
    outcome is associated with observedness conditional on the supplied columns.

    This is an exploratory association screen. It does **not** test or identify MAR
    versus MNAR: those mechanisms are not identifiable from observed data alone.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame whose column-observedness indicators are screened.
    outcome : array-like
        Fully observed auxiliary outcome of length ``len(X)``. Non-numeric values are
        deterministically factorized for the one-degree-of-freedom screening model.
    missing_values : list, optional
        Extra sentinel values treated as missing in ``X``.

    Returns
    -------
    pandas.DataFrame
        Columns are ``feature``, ``lrt_statistic``, ``p_value``, and ``n_obs``.

    Raises
    ------
    TypeError
        If ``X`` is not a DataFrame.
    ValueError
        If ``outcome`` has a different length from ``X`` or contains missing values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"age": [20.0, np.nan, 40.0], "group": [0, 1, 1]})
    >>> list(missingness_association_test(frame, [0, 1, 1]).columns)
    ['feature', 'lrt_statistic', 'p_value', 'n_obs']

    Notes
    -----
    The likelihood-ratio reference distribution is an approximation and this screen is
    not a replacement for a pre-specified missing-data model. Treat p-values as
    exploratory, especially when screening many features.
    """
    validate_dataframe(X, param="X")
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    outcome_series = pd.Series(outcome)
    if len(outcome_series) != len(X):
        raise ValueError("outcome must have the same length as X.")
    if outcome_series.isna().any():
        raise ValueError(
            "outcome must be fully observed for an observedness association screen."
        )
    if pd.api.types.is_numeric_dtype(outcome_series):
        outcome_values = outcome_series.to_numpy(dtype=float)
    else:
        outcome_values = pd.factorize(outcome_series, sort=True)[0].astype(float)

    observedness = (~X.isna()).astype(int)
    results = []
    for feature in X.columns:
        response = observedness[feature].to_numpy()
        if np.all(response == 1) or np.all(response == 0):
            continue

        others = [column for column in X.columns if column != feature]
        predictors = pd.get_dummies(X[others], dummy_na=True, dtype=float)
        baseline_features = predictors.fillna(0.0).to_numpy(dtype=float)
        if baseline_features.shape[1] == 0:
            baseline_features = np.zeros((len(X), 1), dtype=float)
        augmented_features = np.column_stack([baseline_features, outcome_values])

        # ``C=np.inf`` with the default L2 penalty is scikit-learn's supported
        # replacement for the deprecated ``penalty=None`` configuration.
        baseline = LogisticRegression(C=np.inf, fit_intercept=True, solver="lbfgs")
        augmented = LogisticRegression(C=np.inf, fit_intercept=True, solver="lbfgs")
        try:
            baseline.fit(baseline_features, response)
            augmented.fit(augmented_features, response)
        except ValueError:
            continue

        statistic = max(
            2.0 * (
                _logistic_log_likelihood(augmented, augmented_features, response)
                - _logistic_log_likelihood(baseline, baseline_features, response)
            ),
            0.0,
        )
        results.append({
            "feature": feature,
            "lrt_statistic": float(statistic),
            "p_value": float(1.0 - chi2.cdf(statistic, df=1)),
            "n_obs": int(len(response)),
        })

    return pd.DataFrame(
        results,
        columns=["feature", "lrt_statistic", "p_value", "n_obs"],
    )


def mar_mnar_test(
    X: pd.DataFrame,
    Y,
    missing_values: Optional[List] = None,
) -> List:
    """Deprecated wrapper for :func:`missingness_association_test`.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame whose observedness indicators are screened.
    Y : array-like
        Fully observed auxiliary outcome retained for backward compatibility.
    missing_values : list, optional
        Extra sentinel values treated as missing in ``X``.

    Returns
    -------
    list of tuple
        Legacy ``(feature, lrt_statistic, p_value)`` tuples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> mar_mnar_test(pd.DataFrame({"x": [1.0, np.nan, 3.0]}), [0, 1, 1])
    []

    Warnings
    --------
    FutureWarning
        The name is deprecated because observed data cannot identify MAR versus MNAR.
    """
    warnings.warn(
        "mar_mnar_test cannot identify MAR versus MNAR from observed data and is "
        "deprecated; use missingness_association_test for an explicitly limited "
        "observed-data association screen.",
        FutureWarning,
        stacklevel=2,
    )
    result = missingness_association_test(X, Y, missing_values)
    return list(result[["feature", "lrt_statistic", "p_value"]].itertuples(
        index=False,
        name=None,
    ))


def diagnose_missing(
    df: pd.DataFrame,
    significance: float = 0.05,
    missing_values: Optional[List] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    ridge: float = 1e-6,
) -> Dict:
    """Summarise observed-data evidence relevant to a missing-data plan.

    Runs Little's MCAR test when its numeric-data preconditions are met and reports
    missingness prevalence plus descriptive nullity correlation. The result never
    identifies MAR versus MNAR: those mechanisms cannot be distinguished from observed
    data alone.

    Parameters
    ----------
    df : pd.DataFrame
    significance : float, default 0.05
        p-value threshold for the MCAR test.
    missing_values : list, optional
    max_iter : int, default 100
    tol : float, default 1e-5
    ridge : float, default 1e-6

    Returns
    -------
    dict
        All keys from :func:`mcar_test` plus ``mcar_evidence``, legacy-compatible
        ``mechanism``, ``recommendation``, ``strategy_hint``,
        ``high_missingness_cols``, and descriptive ``max_nullity_corr``. The only
        valid values of ``mcar_evidence`` are ``"MCAR_not_rejected"``,
        ``"MCAR_rejected"``, and ``"insufficient_data"``.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     'age':    [25, np.nan, 35, 40, np.nan],
    ...     'income': [50000, 60000, np.nan, 80000, 55000],
    ... })
    >>> result = diagnose_missing(df)
    >>> result['mcar_evidence'] in ('MCAR_not_rejected', 'MCAR_rejected', 'insufficient_data')
    True

    Notes
    -----
    A non-rejection of MCAR is not proof of MCAR. Nullity correlation describes joint
    observedness patterns; it does not identify MAR or MNAR. Choose the imputation and
    analysis model from substantive knowledge, then assess sensitivity to defensible
    missing-data assumptions.
    """
    validate_dataframe(df, param="df")
    if missing_values is not None:
        df = df.replace(missing_values, np.nan)

    miss_pct = df.isnull().mean()
    high_miss_cols: List[str] = miss_pct[miss_pct > 0.40].index.tolist()

    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.loc[:, num_df.isnull().any()]

    max_corr: Optional[float] = None
    if num_df.shape[1] >= 2:
        corr_arr = num_df.isnull().astype(float).corr().abs().to_numpy().copy()
        np.fill_diagonal(corr_arr, 0.0)
        max_corr = float(corr_arr.max())

    test_result: Dict = {
        "chi_square": None,
        "df": None,
        "p_value": None,
        "missing_patterns": None,
        "amount_missing": df.isnull().agg(["sum", "mean"]).rename(
            index={"sum": "Number Missing", "mean": "Percent Missing"}
        ),
    }
    can_test = num_df.shape[1] >= 2 and num_df.shape[0] >= 4

    if can_test:
        try:
            test_result = mcar_test(num_df, max_iter=max_iter, tol=tol, ridge=ridge)
        except (ValueError, RuntimeError, LinAlgError, ArithmeticError):
            can_test = False

    p_val = test_result.get("p_value")

    if not can_test or p_val is None or np.isnan(p_val):
        mcar_evidence = "insufficient_data"
        recommendation = (
            "Not enough numeric columns or rows to run Little's MCAR test. "
            "Document the intended analysis, inspect missingness patterns, and choose "
            "a missing-data assumption from substantive knowledge."
        )
        strategy_hint = "Specify an analysis model and run sensitivity analyses."

    elif p_val >= significance:
        mcar_evidence = "MCAR_not_rejected"
        recommendation = (
            f"Little's test p={p_val:.3f} >= {significance}: MCAR was not rejected. "
            "This is not proof of MCAR and does not justify single-value imputation "
            "for inferential analyses."
            + (
                f" However, {len(high_miss_cols)} column(s) have >40% missingness "
                f"({high_miss_cols}); assess whether the intended analysis remains credible."
                if high_miss_cols else ""
            )
        )
        strategy_hint = "Use multiple imputation or a model-based analysis when justified; run sensitivity analyses."

    else:
        mcar_evidence = "MCAR_rejected"
        recommendation = (
            f"Little's test p={p_val:.3f} < {significance}: evidence against MCAR. "
            "Observed data cannot distinguish MAR from MNAR, so the analysis requires "
            "an explicit assumption and sensitivity analysis."
            + (
                f" Flagged high-missingness columns: {high_miss_cols}."
                if high_miss_cols else ""
            )
        )
        strategy_hint = (
            "Specify a MAR analysis if scientifically defensible and assess MNAR "
            "departures with a sensitivity analysis."
        )

    return {
        **test_result,
        "mcar_evidence": mcar_evidence,
        "mechanism": mcar_evidence,
        "recommendation": recommendation,
        "strategy_hint": strategy_hint,
        "high_missingness_cols": high_miss_cols,
        "max_nullity_corr": max_corr,
    }


# ===========================================================================
# Section 3 — MICE Convergence Diagnostics
# ===========================================================================


def _validate_mice_histories(
    histories: List[Dict[str, List[float]]],
    variables: Optional[List[str]] = None,
) -> List[str]:
    """Validate MICE history payload and return variables to analyse.

    Parameters
    ----------
    histories : list of dict
        One history dict per chain, where each dict maps
        ``column_name -> [mean_iter_1, ..., mean_iter_k]``.
    variables : list of str, optional
        User-requested variables.  If None, all eligible variables are used.

    Returns
    -------
    list of str
        Variables present in at least two chains with at least two iterations.

    Raises
    ------
    ValueError
        If fewer than 2 histories are provided or payload is malformed.
    """
    if len(histories) < 2:
        raise ValueError(
            f"mice_convergence requires at least 2 chains; got {len(histories)}."
        )

    if not all(isinstance(h, dict) for h in histories):
        raise ValueError(
            "Each chain history must be a dict mapping column names to "
            "lists of per-iteration means."
        )

    available: set = set()
    for hist in histories:
        for key, value in hist.items():
            if isinstance(value, list) and len(value) >= 2:
                available.add(key)

    if variables is None:
        return sorted(available)

    selected = [var for var in variables if var in available]
    return selected


def _gelman_rubin_rhat(traces: np.ndarray) -> float:
    """Compute the Gelman-Rubin potential scale reduction factor (R-hat).

    Implements the classical estimator from Gelman & Rubin (1992):
    "Inference from iterative simulation using multiple sequences".
    Statistical Science, 7(4), 457-472.

    Parameters
    ----------
    traces : np.ndarray, shape (m, n)
        Matrix of chain traces where *m* is the number of chains and
        *n* is the number of iterations per chain.  Requires m >= 2
        and n >= 2.

    Returns
    -------
    float
        Potential scale reduction factor.  Values close to 1.0 indicate
        convergence.  R-hat < 1.1 is generally acceptable; R-hat < 1.05
        is preferred.

    Notes
    -----
    When within-chain variance ``W`` is zero (all chains are constant),
    returns 1.0 — technically converged at a single point.
    Returns ``np.nan`` if m < 2 or n < 2.
    """
    m, n = traces.shape
    if m < 2 or n < 2:
        return np.nan

    chain_means = traces.mean(axis=1)          # shape (m,)
    chain_vars = traces.var(axis=1, ddof=1)    # shape (m,)

    B = n * chain_means.var(ddof=1)            # between-chain variance * n
    W = chain_vars.mean()                      # within-chain variance

    if W == 0:
        return 1.0

    var_hat = ((n - 1) / n) * W + (1 / n) * B
    return float(np.sqrt(var_hat / W))


def mice_convergence(
    histories: List[Dict[str, List[float]]],
    variables: Optional[List[str]] = None,
    rhat_threshold: float = 1.1,
) -> Dict:
    """Assess convergence of multiple MICE chains using iteration history.

    This replaces the old snapshot-based approach.  The *histories* argument
    must come from ``impute_mice(..., return_history=True)`` which tracks
    the per-iteration mean of imputed values for each column.

    Parameters
    ----------
    histories : list of dict
        One history dict per chain as returned by
        ``impute_mice(..., n_imputations=m, return_history=True)``.
        Each dict maps ``column_name -> [mean_iter_1, ..., mean_iter_k]``.
    variables : list of str, optional
        Specific variables to diagnose.  If None, all variables found in
        the histories with at least 2 iterations are used.
    rhat_threshold : float, default 1.1
        Threshold below which a variable is considered converged.
        R-hat < 1.1 is the standard criterion (Gelman & Rubin, 1992);
        R-hat < 1.05 is preferred for publication-quality analysis.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"rhat"``: Gelman-Rubin R-hat per variable (dict).
        - ``"trace_data"``: per-chain iteration means per variable (dict of dict).
        - ``"converged_by_variable"``: bool per variable (dict).
        - ``"converged"``: overall convergence flag (bool).
        - ``"n_chains"``: number of chains used (int).
        - ``"n_iter_used"``: minimum iterations used per variable (dict).

    Raises
    ------
    ValueError
        If fewer than 2 histories are provided.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import impute_mice
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
    >>> _, histories = impute_mice(
    ...     df, n_imputations=3, max_iter=5, return_history=True
    ... )
    >>> result = mice_convergence(histories)
    >>> "rhat" in result
    True
    >>> result["n_chains"]
    3

    Notes
    -----
    R-hat is computed using per-iteration means of imputed values across
    chains, analogous to the trace-plot diagnostic in R's ``mice`` package
    (van Buuren & Groothuis-Oudshoorn, 2011, J. Stat. Software 45(3)).
    The Gelman-Rubin statistic compares within-chain to between-chain
    variance (Gelman & Rubin, 1992, Statistical Science 7(4), 457-472).
    """
    vars_to_use = _validate_mice_histories(histories, variables=variables)

    if not vars_to_use:
        return {
            "rhat": {},
            "trace_data": {},
            "converged_by_variable": {},
            "converged": True,
            "n_chains": len(histories),
            "n_iter_used": {},
        }

    rhat_dict: Dict[str, float] = {}
    trace_data: Dict[str, Dict[str, List[float]]] = {}
    converged_by_variable: Dict[str, bool] = {}
    n_iter_used: Dict[str, int] = {}

    for var in vars_to_use:
        chains_for_var = []
        for hist in histories:
            values = hist.get(var, [])
            if isinstance(values, list) and len(values) >= 2:
                chains_for_var.append(np.asarray(values, dtype=float))

        if len(chains_for_var) < 2:
            continue

        min_len = min(len(v) for v in chains_for_var)
        if min_len < 2:
            continue

        aligned = np.vstack([v[:min_len] for v in chains_for_var])  # (m, n)
        rhat = _gelman_rubin_rhat(aligned)

        rhat_dict[var] = rhat
        converged_by_variable[var] = bool(np.isfinite(rhat) and rhat < rhat_threshold)
        n_iter_used[var] = int(min_len)
        trace_data[var] = {
            f"chain_{idx}": aligned[idx].tolist()
            for idx in range(aligned.shape[0])
        }

    converged = all(converged_by_variable.values()) if converged_by_variable else True

    return {
        "rhat": rhat_dict,
        "trace_data": trace_data,
        "converged_by_variable": converged_by_variable,
        "converged": converged,
        "n_chains": len(histories),
        "n_iter_used": n_iter_used,
    }


def plot_mice_convergence(
    convergence_result: Dict,
    variables: Optional[List[str]] = None,
    figsize: tuple = (10, 4),
    rhat_threshold: float = 1.1,
) -> None:
    """Plot MICE convergence trace plots with Gelman-Rubin R-hat annotation.

    Each panel shows per-chain iteration means for one imputed variable,
    with the R-hat value annotated in green (converged) or red (not converged).
    This is analogous to the ``plot()`` method on a ``mids`` object in R's
    ``mice`` package.

    Parameters
    ----------
    convergence_result : dict
        Output from :func:`mice_convergence`.
    variables : list of str, optional
        Variables to plot.  If None, all variables in the result are plotted.
    figsize : tuple, default (10, 4)
        Figure size per variable panel ``(width, height)``.
    rhat_threshold : float, default 1.1
        R-hat threshold for green/red annotation colouring.

    Returns
    -------
    None
        Displays matplotlib figure.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import impute_mice
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
    >>> _, histories = impute_mice(
    ...     df, n_imputations=3, max_iter=5, return_history=True
    ... )
    >>> result = mice_convergence(histories)
    >>> plot_mice_convergence(result)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    trace_map = convergence_result.get("trace_data", {})
    rhat_map = convergence_result.get("rhat", {})

    if variables is None:
        variables = list(trace_map.keys())

    if not variables:
        print("No variables to plot.")
        return

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars))

    if n_vars == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        trace = trace_map.get(var, {})
        for chain_name, values in trace.items():
            ax.plot(
                range(1, len(values) + 1),
                values,
                label=chain_name,
                alpha=0.8,
            )

        rhat = rhat_map.get(var, np.nan)
        color = "green" if (np.isfinite(rhat) and rhat < rhat_threshold) else "red"
        rhat_label = f"R\u0302 = {rhat:.3f}" if np.isfinite(rhat) else "R\u0302 = nan"

        ax.set_title(f"MICE Trace Plot: {var}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean imputed value")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.02,
            0.95,
            rhat_label,
            transform=ax.transAxes,
            verticalalignment="top",
            color=color,
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
