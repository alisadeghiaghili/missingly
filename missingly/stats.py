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

import numpy as np
import pandas as pd
from numpy.linalg import inv, slogdet
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

def _em_mle_estimation(data, max_iter=100, tol=1e-5, ridge=1e-6):
    """Estimate the mean and covariance matrix of a dataset with missing values
    using a basic EM algorithm under the assumption of multivariate normality.
    """
    X = data.copy()
    n, d = X.shape

    # Initialize missing entries by column means
    col_means = np.nanmean(X, axis=0)
    for j in range(d):
        missing_idx = np.isnan(X[:, j])
        X[missing_idx, j] = col_means[j]

    # Initial estimates
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
        # E-step
        for i in range(n):
            row = data[i, :]
            missing = np.isnan(row)
            if not np.any(missing):
                continue

            observed = ~missing
            mu_obs = mu_hat[observed]
            mu_mis = mu_hat[missing]

            Sigma_oo = Sigma_hat[np.ix_(observed, observed)]
            Sigma_om = Sigma_hat[np.ix_(observed, missing)]
            Sigma_mo = Sigma_hat[np.ix_(missing, observed)]

            Sigma_oo_ridge = Sigma_oo + np.eye(Sigma_oo.shape[0]) * ridge
            inv_Sigma_oo = inv(Sigma_oo_ridge)

            row_obs = row[observed]
            cond_mean = mu_mis + Sigma_mo @ inv_Sigma_oo @ (row_obs - mu_obs)

            X[i, missing] = cond_mean

        # M-step
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


def mcar_test(X: pd.DataFrame, max_iter=100, tol=1e-5, ridge=1e-6, missing_values: list = None):
    """Perform Little's MCAR test for a DataFrame with missing values.

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe to test.
    max_iter : int, optional
        Maximum number of EM iterations.
    tol : float, optional
        Convergence threshold for the EM algorithm.
    ridge : float, optional
        Ridge term for numerical stability.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    dict
        A dictionary containing the chi-square statistic, degrees of freedom,
        p-value, number of missing patterns, and a summary of missing values.
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

    mu_hat, Sigma_hat = _em_mle_estimation(data_np, max_iter=max_iter, tol=tol, ridge=ridge)

    pattern_map = {pat_val: i + 1 for i, pat_val in enumerate(unique_patterns)}
    df_with_pattern["MisPat"] = df_with_pattern["MisPat"].map(pattern_map)

    datasets = {}
    for pat_id in range(1, n_mis_pat + 1):
        subset_df = df_with_pattern[df_with_pattern["MisPat"] == pat_id].iloc[:, :d]
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

    if df_val <= 0:
        p_val = np.nan
    else:
        p_val = 1 - chi2.cdf(d2, df_val)

    amount_missing = pd.DataFrame(
        [nmis, nmis / n],
        index=["Number Missing", "Percent Missing"],
        columns=X.columns
    )

    results = {
        "chi_square": d2,
        "df": df_val,
        "p_value": p_val,
        "missing_patterns": n_mis_pat,
        "amount_missing": amount_missing,
    }
    return results


def _logistic_log_likelihood(model, X, y):
    """Compute the log-likelihood of a fitted LogisticRegression model."""
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)  # Avoid log(0)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll


def mar_mnar_test(X, Y, missing_values: list = None):
    """Perform likelihood ratio tests for MAR vs. MNAR.

    This test compares two models for each feature with missing values:
    - MAR model: Missingness ~ other features
    - MNAR model: Missingness ~ other features + outcome variable

    Parameters
    ----------
    X : pd.DataFrame
        The covariate matrix with missing values.
    Y : np.ndarray
        The outcome vector.
    missing_values : list, optional
        A list of values to be considered as missing.

    Returns
    -------
    list
        A list of tuples, where each tuple contains the feature name,
        the LRT statistic, and the p-value.
    """
    if missing_values is not None:
        X = X.replace(missing_values, np.nan)
        
    # Binary missingness indicators
    D_matrix = (~X.isna()).astype(int)

    results = []
    for feature in X.columns:
        D = D_matrix[feature].values

        # If fully observed or fully missing, skip
        if np.all(D == 1) or np.all(D == 0):
            continue

        # Build MAR model (missingness ~ other features)
        other_feats = [col for col in X.columns if col != feature]
        # Simple imputation of 0 for missing in other features
        X_other = X[other_feats].fillna(0).values

        # Fit logistic model for MAR
        mar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mar_model.fit(X_other, D)

        # Build MNAR model (missingness ~ other features + Y)
        X_mnar = np.column_stack([X_other, Y])
        mnar_model = LogisticRegression(penalty=None, fit_intercept=True, solver='lbfgs')
        mnar_model.fit(X_mnar, D)

        # Compute log-likelihoods
        ll_mar = _logistic_log_likelihood(mar_model, X_other, D)
        ll_mnar = _logistic_log_likelihood(mnar_model, X_mnar, D)

        # LRT statistic
        LRT_statistic = 2 * (ll_mnar - ll_mar)
        # Degrees of freedom = 1 (added Y)
        p_value = 1 - chi2.cdf(LRT_statistic, df=1)

        results.append((feature, LRT_statistic, p_value))
    return results
