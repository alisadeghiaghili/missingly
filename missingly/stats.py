# This code is adapted from the XeroGraph library by Julhash Kazi
# Original source: https://github.com/kazilab/XeroGraph
# The XeroGraph library is licensed under the Apache License 2.0.
#
# Copyright 2024 Julhash Kazi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Statistical tests and diagnostics for missing data mechanisms.

Public API
----------
mcar_test
    Little's MCAR test — returns chi-square statistic, p-value, and
    per-column missingness summary.
mar_mnar_test
    Likelihood-ratio test distinguishing MAR from MNAR.
diagnose_missing
    Actionable wrapper around :func:`mcar_test` that returns a plain-
    English ``recommendation`` and ``strategy_hint`` alongside the raw
    test results, so callers don't have to interpret p-values themselves.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


def _em_mle_estimation(data, max_iter=100, tol=1e-5, ridge=1e-6):
    """Estimate mean and covariance of a dataset with missing values via EM.

    Uses a basic EM algorithm under the assumption of multivariate
    normality.  Missing entries are initialised with column means then
    iteratively refined.

    Parameters
    ----------
    data : np.ndarray, shape (n, d)
        Raw numeric array; NaN marks missing values.
    max_iter : int
        Maximum EM iterations.  Default 100.
    tol : float
        Log-likelihood convergence threshold.  Default 1e-5.
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
        missing_idx = np.isnan(X[:, j])
        X[missing_idx, j] = col_means[j]

    mu_hat = np.mean(X, axis=0)
    Sigma_hat = np.cov(X, rowvar=False)

    def log_likelihood(xx, mu, Sigma):
        ddim = xx.shape[1]
        sign, logdet_val = slogdet(Sigma)
        if sign <= 0 or np.isinf(logdet_val):
            return -np.inf
        inv_Sigma = inv(Sigma)
        const = 0.5 * (ddim * np.log(2 * np.pi) + logdet_val)
        ll = 0.0
        for i in range(n):
            diff = xx[i] - mu
            ll -= 0.5 * diff @ inv_Sigma @ diff
        return ll - n * const

    old_ll = -np.inf
    for _ in range(max_iter):
        for i in range(n):
            row = data[i, :]
            missing = np.isnan(row)
            if not np.any(missing):
                continue
            observed = ~missing
            mu_obs = mu_hat[observed]
            mu_mis = mu_hat[missing]
            Sigma_oo = Sigma_hat[np.ix_(observed, observed)]
            Sigma_mo = Sigma_hat[np.ix_(missing, observed)]
            Sigma_oo_ridge = Sigma_oo + np.eye(Sigma_oo.shape[0]) * ridge
            inv_Sigma_oo = inv(Sigma_oo_ridge)
            row_obs = row[observed]
            cond_mean = mu_mis + Sigma_mo @ inv_Sigma_oo @ (row_obs - mu_obs)
            X[i, missing] = cond_mean

        mu_new = np.mean(X, axis=0)
        Sigma_new = np.cov(X, rowvar=False)
        Sigma_new += np.eye(d) * ridge
        ll_new = log_likelihood(X, mu_new, Sigma_new)
        if abs(ll_new - old_ll) < tol:
            mu_hat, Sigma_hat = mu_new, Sigma_new
            break
        mu_hat, Sigma_hat = mu_new, Sigma_new
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
    (MCAR).  A small p-value (typically < 0.05) gives evidence *against*
    MCAR.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with missing values.  Must contain at least two numeric
        columns.
    max_iter : int, optional
        Maximum EM iterations.  Default 100.
    tol : float, optional
        EM convergence threshold.  Default 1e-5.
    ridge : float, optional
        Diagonal ridge for numerical stability.  Default 1e-6.
    missing_values : list, optional
        Extra sentinel values to treat as missing.

    Returns
    -------
    dict
        Keys: ``chi_square``, ``df``, ``p_value``,
        ``missing_patterns``, ``amount_missing``.
    """
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    data_np = X.to_numpy(dtype=float)
    n, d = data_np.shape

    r = np.isnan(data_np).astype(int)
    nmis = r.sum(axis=0)

    powers_of_2 = 2 ** np.arange(d)
    mdp = r.dot(powers_of_2) + 1

    df_with_pattern = pd.DataFrame(data_np.copy(), columns=X.columns)
    df_with_pattern["MisPat"] = mdp

    unique_patterns = np.sort(df_with_pattern["MisPat"].unique())
    n_mis_pat = len(unique_patterns)

    mu_hat, Sigma_hat = _em_mle_estimation(
        data_np, max_iter=max_iter, tol=tol, ridge=ridge
    )

    pattern_map = {pat_val: i + 1 for i, pat_val in enumerate(unique_patterns)}
    df_with_pattern["MisPat"] = df_with_pattern["MisPat"].map(pattern_map)

    datasets = {}
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = df_with_pattern[df_with_pattern["MisPat"] == pat_id].iloc[
            :, :d
        ]
        datasets[pat_id] = subset_df

    df_val = 0
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = datasets[pat_id]
        col_sums = subset_df.isna().sum(axis=0)
        observed_count = (col_sums == 0).sum()
        df_val += observed_count
    df_val -= d

    d2 = 0.0
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = datasets[pat_id]
        n_pat = subset_df.shape[0]
        pat_mean = subset_df.mean(axis=0).values
        col_sums = subset_df.isna().sum(axis=0)
        obs_cols = np.where(col_sums == 0)[0]
        if len(obs_cols) == 0:
            continue
        mean_diff = pat_mean[obs_cols] - mu_hat[obs_cols]
        Sigma_obs = Sigma_hat[np.ix_(obs_cols, obs_cols)]
        Sigma_obs += np.eye(Sigma_obs.shape[0]) * ridge
        Sigma_obs_inv = inv(Sigma_obs)
        diff_vec = mean_diff.reshape(-1, 1)
        contrib = n_pat * (diff_vec.T @ Sigma_obs_inv @ diff_vec)
        d2 += contrib[0, 0]

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


def diagnose_missing(
    df: pd.DataFrame,
    significance: float = 0.05,
    missing_values: Optional[List] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    ridge: float = 1e-6,
) -> Dict:
    """Diagnose the missing data mechanism and return actionable recommendations.

    This is the opinionated, actionable wrapper around :func:`mcar_test`.
    It runs the statistical test, inspects nullity correlations, and
    returns a plain-English ``recommendation`` and a ``strategy_hint``
    that directly maps to missingly imputation functions.

    Decision logic
    --------------
    1. Run Little's MCAR test on numeric columns.
    2. If p >= *significance* → data are consistent with MCAR.
       Simple imputers (mean/median/mode) are appropriate.
    3. If p < *significance* → evidence against MCAR.  Inspect the
       maximum absolute nullity correlation between column pairs:
       - If max_corr > 0.3 → structured MAR: use model-based imputers
         (KNN, MICE, RF) that exploit inter-column relationships.
       - If max_corr <= 0.3 → possible MNAR or weak MAR: flag the
         specific columns with high missingness and recommend domain
         review before imputing with model-based methods.
    4. Columns with > 40 % missingness are always flagged regardless of
       mechanism, because any imputer will introduce substantial bias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to diagnose.  Non-numeric columns are excluded from
        Little's test but included in the high-missingness flag.
    significance : float, optional
        p-value threshold for the MCAR test.  Default 0.05.
    missing_values : list, optional
        Extra sentinel values treated as missing.
    max_iter : int, optional
        Passed to :func:`mcar_test`.  Default 100.
    tol : float, optional
        Passed to :func:`mcar_test`.  Default 1e-5.
    ridge : float, optional
        Passed to :func:`mcar_test`.  Default 1e-6.

    Returns
    -------
    dict
        All keys from :func:`mcar_test` plus:

        ``mechanism`` : str
            One of ``"MCAR"``, ``"MAR"``, ``"possible_MNAR"``,
            ``"insufficient_data"``.
        ``recommendation`` : str
            Plain-English explanation of the diagnosis.
        ``strategy_hint`` : str
            Suggested missingly imputation function(s) to call.
        ``high_missingness_cols`` : list[str]
            Columns where > 40 % of values are missing.
        ``max_nullity_corr`` : float or None
            Maximum absolute pairwise nullity correlation (numeric cols
            only).  ``None`` when fewer than 2 numeric columns exist.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     'age':    [25, np.nan, 35, 40, np.nan],
    ...     'income': [50000, 60000, np.nan, 80000, 55000],
    ...     'score':  [7.5, 8.0, np.nan, 9.0, 7.0],
    ... })
    >>> result = diagnose_missing(df)
    >>> print(result['mechanism'])
    >>> print(result['recommendation'])
    >>> print(result['strategy_hint'])
    """
    if missing_values is not None:
        df = df.replace(missing_values, np.nan)

    # --- high-missingness flag (all columns) ---
    miss_pct = df.isnull().mean()
    high_miss_cols: List[str] = miss_pct[miss_pct > 0.40].index.tolist()

    # --- limit to numeric for Little's test ---
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.loc[:, num_df.isnull().any()]  # keep only cols with missing

    # --- nullity correlation ---
    max_corr: Optional[float] = None
    if num_df.shape[1] >= 2:
        null_corr = num_df.isnull().astype(float).corr().abs()
        # zero out diagonal
        np.fill_diagonal(null_corr.values, 0.0)
        max_corr = float(null_corr.values.max())

    # --- run Little's test if possible ---
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
            test_result = mcar_test(
                num_df, max_iter=max_iter, tol=tol, ridge=ridge
            )
        except Exception:  # pragma: no cover — numerical edge-cases
            can_test = False

    p_val = test_result.get("p_value")

    # --- decision logic ---
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
            f"Little's test p={p_val:.3f} ≥ {significance}: data are consistent "
            f"with Missing Completely At Random (MCAR). "
            f"Simple imputers (mean, median, mode) introduce minimal bias. "
            + (
                f" However, {len(high_miss_cols)} column(s) have >40% missingness "
                f"({high_miss_cols}); consider dropping them instead of imputing."
                if high_miss_cols
                else ""
            )
        )
        strategy_hint = "impute_mean() / impute_median() / impute_mode()"

    elif max_corr is not None and max_corr > 0.30:
        mechanism = "MAR"
        recommendation = (
            f"Little's test p={p_val:.3f} < {significance}: evidence against MCAR. "
            f"Maximum nullity correlation between column pairs is {max_corr:.2f} > 0.30, "
            f"suggesting Missing At Random (MAR): missingness depends on other observed "
            f"columns. Use model-based imputers that exploit inter-column structure."
            + (
                f" Caution: {len(high_miss_cols)} column(s) have >40% missingness "
                f"({high_miss_cols}) — even good imputers will introduce bias here."
                if high_miss_cols
                else ""
            )
        )
        strategy_hint = (
            "impute_knn() for moderate datasets; "
            "impute_mice() for multivariate structure; "
            "impute_rf() / impute_gb() for non-linear relationships. "
            "Use MissinglyImputer inside a Pipeline to avoid data leakage."
        )

    else:
        mechanism = "possible_MNAR"
        recommendation = (
            f"Little's test p={p_val:.3f} < {significance}: evidence against MCAR. "
            f"Maximum nullity correlation is "
            + (
                f"{max_corr:.2f} ≤ 0.30" if max_corr is not None else "unavailable"
            )
            + ", so missingness does not strongly correlate with other observed "
            f"columns. This is consistent with Missing Not At Random (MNAR): "
            f"the probability of being missing depends on the unobserved value "
            f"itself (e.g. high-earners skip the income field). "
            f"Standard imputers cannot correct MNAR bias without external "
            f"assumptions. Options: (1) collect more data, (2) use "
            f"domain knowledge to bound the missing values, (3) sensitivity "
            f"analysis with multiple imputation strategies."
            + (
                f" Flagged high-missingness columns: {high_miss_cols}."
                if high_miss_cols
                else ""
            )
        )
        strategy_hint = (
            "Domain review required. If you must impute: "
            "impute_mice() with multiple seeds for sensitivity analysis. "
            "Use MissinglyImputer inside a Pipeline to avoid data leakage."
        )

    return {
        **test_result,
        "mechanism": mechanism,
        "recommendation": recommendation,
        "strategy_hint": strategy_hint,
        "high_missingness_cols": high_miss_cols,
        "max_nullity_corr": max_corr,
    }


def _logistic_log_likelihood(model, X, y):
    """Compute the log-likelihood of a fitted LogisticRegression model.

    Parameters
    ----------
    model : fitted LogisticRegression
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target.

    Returns
    -------
    float
    """
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mar_mnar_test(
    X: pd.DataFrame,
    Y,
    missing_values: Optional[List] = None,
) -> List:
    """Likelihood-ratio test distinguishing MAR from MNAR per feature.

    Compares two logistic models for each feature with missing values:

    * MAR model  : ``missingness_indicator ~ other_features``
    * MNAR model : ``missingness_indicator ~ other_features + Y``

    A significant LRT statistic (small p-value) means including *Y*
    improves the fit, i.e. missingness in that feature is related to the
    outcome — evidence for MNAR.

    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix with missing values.
    Y : array-like
        Outcome vector (binary or continuous).
    missing_values : list, optional
        Extra sentinel values treated as missing.

    Returns
    -------
    list of tuple
        Each tuple is ``(feature_name, LRT_statistic, p_value)``.
    """
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)

    D_matrix = (~X.isna()).astype(int)
    results = []

    for feature in X.columns:
        D = D_matrix[feature].values
        if np.all(D == 1) or np.all(D == 0):
            continue

        other_feats = [col for col in X.columns if col != feature]
        X_other = X[other_feats].fillna(0).values

        mar_model = LogisticRegression(
            penalty=None, fit_intercept=True, solver="lbfgs"
        )
        mar_model.fit(X_other, D)

        X_mnar = np.column_stack([X_other, Y])
        mnar_model = LogisticRegression(
            penalty=None, fit_intercept=True, solver="lbfgs"
        )
        mnar_model.fit(X_mnar, D)

        ll_mar = _logistic_log_likelihood(mar_model, X_other, D)
        ll_mnar = _logistic_log_likelihood(mnar_model, X_mnar, D)

        LRT_statistic = 2 * (ll_mnar - ll_mar)
        p_value = float(1 - chi2.cdf(LRT_statistic, df=1))
        results.append((feature, LRT_statistic, p_value))

    return results
