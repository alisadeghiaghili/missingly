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
mar_mnar_test
    Likelihood-ratio test distinguishing MAR from MNAR.
diagnose_missing
    Actionable wrapper: returns mechanism, recommendation, strategy_hint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
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

def _em_mle_estimation(data, max_iter=100, tol=1e-5, ridge=1e-6):
    """Estimate mean and covariance via EM for multivariate-normal data with NaN.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Raw numeric array; NaN marks missing values.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence threshold.
    ridge : float
        Diagonal ridge added to covariance for numerical stability.

    Returns
    -------
    mu_hat : np.ndarray, shape (d,)
    Sigma_hat : np.ndarray, shape (d, d)
    """
    X = data.copy()
    n, d = X.shape

    col_means = np.nanmean(X, axis=0)
    for j in range(d):
        X[np.isnan(X[:, j]), j] = col_means[j]

    mu_hat = np.mean(X, axis=0)
    Sigma_hat = np.cov(X, rowvar=False)

    def _log_likelihood(xx, mu, Sigma):
        sign, logdet_val = slogdet(Sigma)
        if sign <= 0 or np.isinf(logdet_val):
            return -np.inf
        inv_S = inv(Sigma)
        const = 0.5 * (d * np.log(2 * np.pi) + logdet_val)
        ll = sum(
            -0.5 * (xx[i] - mu) @ inv_S @ (xx[i] - mu)
            for i in range(n)
        )
        return ll - n * const

    old_ll = -np.inf
    for _ in range(max_iter):
        for i in range(n):
            row = data[i, :]
            missing = np.isnan(row)
            if not missing.any():
                continue
            observed = ~missing
            Sigma_oo = Sigma_hat[np.ix_(observed, observed)] + np.eye(observed.sum()) * ridge
            Sigma_mo = Sigma_hat[np.ix_(missing, observed)]
            cond_mean = (
                mu_hat[missing]
                + Sigma_mo @ inv(Sigma_oo) @ (row[observed] - mu_hat[observed])
            )
            X[i, missing] = cond_mean

        mu_hat = np.mean(X, axis=0)
        Sigma_hat = np.cov(X, rowvar=False) + np.eye(d) * ridge
        ll_new = _log_likelihood(X, mu_hat, Sigma_hat)
        if abs(ll_new - old_ll) < tol:
            break
        old_ll = ll_new

    return mu_hat, Sigma_hat


def mcar_test(
    X: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-5,
    ridge: float = 1e-6,
    missing_values: Optional[List] = None,
) -> Dict:
    """Perform Little's MCAR test.

    Tests the null hypothesis that data are Missing Completely At Random
    (MCAR).  A small p-value (< 0.05) gives evidence *against* MCAR.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with missing values.  Must contain at least two numeric
        columns.
    max_iter : int, default 100
        Maximum EM iterations.
    tol : float, default 1e-5
        EM convergence threshold.
    ridge : float, default 1e-6
        Diagonal ridge for numerical stability.
    missing_values : list, optional
        Extra sentinel values treated as missing.

    Returns
    -------
    dict
        Keys: ``chi_square``, ``df``, ``p_value``,
        ``missing_patterns``, ``amount_missing``.

    Raises
    ------
    TypeError
        If *X* is not a :class:`pandas.DataFrame`.
    ValueError
        If *X* is empty.
    """
    validate_dataframe(X, param="X")
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    # Check for missing values
    if not X.isnull().any(axis=None):
        raise ValueError(
            "mcar_test requires at least one missing value in the DataFrame. "
            "All values are present — MCAR test is not applicable."
        )

    data_np = X.to_numpy(dtype=float)
    n, d = data_np.shape

    r = np.isnan(data_np).astype(int)
    nmis = r.sum(axis=0)
    powers_of_2 = 2 ** np.arange(d)
    mdp = r.dot(powers_of_2) + 1

    df_wp = pd.DataFrame(data_np.copy(), columns=X.columns)
    df_wp["MisPat"] = mdp

    unique_patterns = np.sort(df_wp["MisPat"].unique())
    n_mis_pat = len(unique_patterns)

    mu_hat, Sigma_hat = _em_mle_estimation(
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
        Sigma_obs = Sigma_hat[np.ix_(obs_cols, obs_cols)] + np.eye(len(obs_cols)) * ridge
        d2 += n_pat * (mean_diff @ inv(Sigma_obs) @ mean_diff)

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
    }


def _logistic_log_likelihood(model, X, y):
    """Log-likelihood of a fitted LogisticRegression."""
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mar_mnar_test(
    X: pd.DataFrame,
    Y,
    missing_values: Optional[List] = None,
) -> List:
    """Likelihood-ratio test distinguishing MAR from MNAR per feature.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor DataFrame.
    Y : array-like
        Outcome variable (used only in the MNAR model).
    missing_values : list, optional

    Returns
    -------
    list of tuple
        Each tuple is ``(feature_name, LRT_statistic, p_value)``.

    Raises
    ------
    TypeError
        If *X* is not a :class:`pandas.DataFrame`.
    """
    validate_dataframe(X, param="X")
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    D_matrix = (~X.isna()).astype(int)
    results = []

    for feature in X.columns:
        D = D_matrix[feature].values
        if np.all(D == 1) or np.all(D == 0):
            continue

        other_feats = [c for c in X.columns if c != feature]
        X_other = X[other_feats].fillna(0).values

        mar_model = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
        mar_model.fit(X_other, D)

        mnar_model = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
        mnar_model.fit(np.column_stack([X_other, Y]), D)

        LRT = 2 * (
            _logistic_log_likelihood(mnar_model, np.column_stack([X_other, Y]), D)
            - _logistic_log_likelihood(mar_model, X_other, D)
        )
        results.append((feature, LRT, float(1 - chi2.cdf(LRT, df=1))))

    return results


def diagnose_missing(
    df: pd.DataFrame,
    significance: float = 0.05,
    missing_values: Optional[List] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    ridge: float = 1e-6,
) -> Dict:
    """Diagnose the missing data mechanism and return actionable recommendations.

    Runs Little's MCAR test, inspects nullity correlations, and returns a
    plain-English ``recommendation`` and a ``strategy_hint`` that maps
    directly to missingly imputation functions.

    Decision logic
    --------------
    1. Run Little's MCAR test on numeric columns.
    2. p \u2265 *significance* \u2192 MCAR: simple imputers are appropriate.
    3. p < *significance* \u2192 not MCAR.  Check max nullity correlation:
       - > 0.30 \u2192 MAR: use model-based imputers (KNN, MICE, RF).
       - \u2264 0.30 \u2192 possible MNAR: flag and recommend domain review.
    4. Columns with > 40% missingness are always flagged.

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
        All keys from :func:`mcar_test` plus:
        ``mechanism``, ``recommendation``, ``strategy_hint``,
        ``high_missingness_cols``, ``max_nullity_corr``.

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
    >>> result['mechanism'] in ('MCAR', 'MAR', 'possible_MNAR', 'insufficient_data')
    True
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
        except (ValueError, np.linalg.LinAlgError, ArithmeticError):  # pragma: no cover — numerical edge-cases
            can_test = False

    p_val = test_result.get("p_value")

    if not can_test or p_val is None or np.isnan(p_val):
        mechanism = "insufficient_data"
        recommendation = (
            "Not enough numeric columns or rows to run Little's MCAR test. "
            "Inspect the data manually. For small datasets, use "
            "impute_mean or impute_median as a safe default."
        )
        strategy_hint = "impute_mean() or impute_median()"

    elif p_val >= significance:
        mechanism = "MCAR"
        recommendation = (
            f"Little's test p={p_val:.3f} \u2265 {significance}: data are consistent "
            "with Missing Completely At Random (MCAR). "
            "Simple imputers (mean, median, mode) introduce minimal bias."
            + (
                f" However, {len(high_miss_cols)} column(s) have >40% missingness "
                f"({high_miss_cols}); consider dropping them instead of imputing."
                if high_miss_cols else ""
            )
        )
        strategy_hint = "impute_mean() / impute_median() / impute_mode()"

    elif max_corr is not None and max_corr > 0.30:
        mechanism = "MAR"
        recommendation = (
            f"Little's test p={p_val:.3f} < {significance}: evidence against MCAR. "
            f"Maximum nullity correlation is {max_corr:.2f} > 0.30, "
            "suggesting MAR. Use model-based imputers."
            + (
                f" Caution: {len(high_miss_cols)} column(s) have >40% missingness "
                f"({high_miss_cols})."
                if high_miss_cols else ""
            )
        )
        strategy_hint = (
            "impute_knn() for moderate datasets; "
            "impute_mice() for multivariate structure; "
            "impute_rf() / impute_gb() for non-linear relationships."
        )

    else:
        corr_str = f"{max_corr:.2f} \u2264 0.30" if max_corr is not None else "unavailable"
        mechanism = "possible_MNAR"
        recommendation = (
            f"Little's test p={p_val:.3f} < {significance}: evidence against MCAR. "
            f"Maximum nullity correlation is {corr_str}, consistent with MNAR. "
            "Standard imputers cannot correct MNAR bias without external assumptions."
            + (
                f" Flagged high-missingness columns: {high_miss_cols}."
                if high_miss_cols else ""
            )
        )
        strategy_hint = (
            "Domain review required. If you must impute: "
            "impute_mice() with multiple seeds for sensitivity analysis."
        )

    return {
        **test_result,
        "mechanism": mechanism,
        "recommendation": recommendation,
        "strategy_hint": strategy_hint,
        "high_missingness_cols": high_miss_cols,
        "max_nullity_corr": max_corr,
    }


# ===========================================================================
# Section 3 — MICE Convergence Diagnostics
# ===========================================================================


def mice_convergence(
    chains: List[pd.DataFrame],
    original_df: pd.DataFrame,
    variables: Optional[List[str]] = None,
) -> Dict:
    """Assess convergence of multiple imputation chains.

    This function implements convergence diagnostics similar to R's mice
    package, including trace plots, R-hat statistics, and density comparison.

    Parameters
    ----------
    chains : list of pd.DataFrame
        List of *m* imputed DataFrames from independent MICE chains.
    original_df : pd.DataFrame
        The original DataFrame with missing values (before imputation).
    variables : list of str, optional
        Specific variables to diagnose. If None, all numeric variables
        with missing values are included.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"rhat"``: Gelman-Rubin R-hat statistic for each variable
        - ``"trace_data"``: Data for trace plots
        - ``"density_data"``: Data for density comparison plots
        - ``"converged"``: Boolean indicating if all R-hat < 1.1

    Raises
    ------
    ValueError
        If fewer than 2 chains are provided.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import impute_mice
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
    >>> chains = [impute_mice(df, random_state=i) for i in range(3)]
    >>> result = mice_convergence(chains, df)
    >>> "rhat" in result
    True

    Notes
    -----
    The Gelman-Rubin R-hat statistic compares between-chain variance to
    within-chain variance. Values close to 1.0 indicate convergence.
    R-hat < 1.1 is generally considered acceptable; R-hat < 1.05 is preferred.
    """
    if len(chains) < 2:
        raise ValueError(
            f"mice_convergence requires at least 2 chains; got {len(chains)}."
        )

    m = len(chains)

    # Determine which variables to diagnose
    if variables is None:
        variables = []
        for col in original_df.columns:
            if original_df[col].isna().any() and pd.api.types.is_numeric_dtype(original_df[col]):
                variables.append(col)

    if not variables:
        return {
            "rhat": {},
            "trace_data": {},
            "density_data": {},
            "converged": True,
        }

    rhat_dict = {}
    trace_data = {}
    density_data = {}

    for var in variables:
        # Extract imputed values for this variable across chains
        chain_values = []
        for chain in chains:
            if var in chain.columns:
                imputed_mask = original_df[var].isna()
                chain_values.append(chain.loc[imputed_mask, var].values)

        if not chain_values or any(len(v) == 0 for v in chain_values):
            continue

        # Ensure all chains have same length
        min_len = min(len(v) for v in chain_values)
        chain_values = [v[:min_len] for v in chain_values]

        # Calculate R-hat
        chain_means = np.array([np.mean(v) for v in chain_values])
        chain_vars = np.array([np.var(v, ddof=1) for v in chain_values])

        # Between-chain variance
        B = np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean(chain_vars)

        # Pooled variance estimate
        n = min_len
        var_plus = ((n - 1) / n) * W + (1 / n) * B

        # R-hat
        rhat = np.sqrt(var_plus / W) if W > 0 else 1.0
        rhat_dict[var] = float(rhat)

        # Trace data: chain index vs imputed value
        trace_data[var] = {
            f"chain_{i}": values.tolist()
            for i, values in enumerate(chain_values)
        }

        # Density data: histogram of imputed values
        all_values = np.concatenate(chain_values)
        hist_counts, hist_edges = np.histogram(all_values, bins=30)
        density_data[var] = {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        }

    converged = all(r < 1.1 for r in rhat_dict.values()) if rhat_dict else True

    return {
        "rhat": rhat_dict,
        "trace_data": trace_data,
        "density_data": density_data,
        "converged": converged,
    }


def plot_mice_convergence(
    convergence_result: Dict,
    variables: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
) -> None:
    """Plot MICE convergence diagnostics.

    Generates trace plots and density comparison plots for visual
    assessment of MICE chain convergence.

    Parameters
    ----------
    convergence_result : dict
        Output from :func:`mice_convergence`.
    variables : list of str, optional
        Variables to plot. If None, all variables in the result are plotted.
    figsize : tuple, default (12, 8)
        Figure size for each variable's diagnostic plot.

    Returns
    -------
    None
        Displays the plots.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import impute_mice
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
    >>> chains = [impute_mice(df, random_state=i) for i in range(3)]
    >>> result = mice_convergence(chains, df)
    >>> plot_mice_convergence(result)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if variables is None:
        variables = list(convergence_result.get("trace_data", {}).keys())

    n_vars = len(variables)
    if n_vars == 0:
        print("No variables to plot.")
        return

    fig, axes = plt.subplots(n_vars, 2, figsize=(figsize[0], figsize[1] * n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for i, var in enumerate(variables):
        # Trace plot
        ax_trace = axes[i, 0]
        trace = convergence_result.get("trace_data", {}).get(var, {})
        for chain_name, values in trace.items():
            ax_trace.plot(values, alpha=0.7, label=chain_name)
        ax_trace.set_title(f"Trace Plot: {var}")
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_ylabel("Value")
        ax_trace.legend(fontsize=8)

        # Density plot
        ax_density = axes[i, 1]
        density = convergence_result.get("density_data", {}).get(var, {})
        if density:
            counts = np.array(density["counts"])
            edges = np.array(density["edges"])
            centers = (edges[:-1] + edges[1:]) / 2
            ax_density.bar(centers, counts, width=edges[1] - edges[0], alpha=0.7)
        ax_density.set_title(f"Imputed Values Distribution: {var}")
        ax_density.set_xlabel("Value")
        ax_density.set_ylabel("Count")

        # Add R-hat annotation
        rhat = convergence_result.get("rhat", {}).get(var, None)
        if rhat is not None:
            color = "green" if rhat < 1.1 else "red"
            axes[i, 0].text(
                0.02, 0.98,
                f"R-hat = {rhat:.3f}",
                transform=axes[i, 0].transAxes,
                verticalalignment="top",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()
