"""Multiple Imputation pooling utilities — Rubin's Rules.

This module implements Rubin's combining rules for scalar estimates and
linear regression coefficients obtained from Multiple Imputation (MI)
workflows.  Typical usage::

    dfs = impute_mice(df, n_imputations=5)
    coefs = [fit_ols(d).params for d in dfs]
    variances = [fit_ols(d).bse**2 for d in dfs]
    result = pool_scalar_estimates(coefs_x, variances_x)

References
----------
* Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
  Wiley.
* van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
  CRC/Taylor & Francis. https://stefvanbuuren.name/fimd/
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["pool_scalar_estimates", "pool_linear_regression_results"]


def pool_scalar_estimates(
    estimates: Sequence[float],
    variances: Sequence[float],
) -> dict:
    """Pool scalar MI estimates using Rubin's combining rules.

    Applies the classic Rubin (1987) formulas to combine *m* point
    estimates and their within-imputation variances into a single pooled
    estimate, total variance, and associated degrees of freedom.

    Parameters
    ----------
    estimates : sequence of float
        Point estimates from each of the *m* imputed datasets,
        e.g. regression coefficients, means, or proportions.
        Length must equal ``len(variances)``.
    variances : sequence of float
        Sampling variance (squared standard error) of each estimate,
        one per imputed dataset.  Must be non-negative.

    Returns
    -------
    dict with keys
        ``q_bar`` : float
            Pooled point estimate (mean of *m* estimates).
        ``u_bar`` : float
            Within-imputation variance (mean of *m* variances).
        ``b`` : float
            Between-imputation variance.
        ``t`` : float
            Total variance = ``u_bar + (1 + 1/m) * b``.
        ``df`` : float
            Barnard–Rubin degrees of freedom for the *t*-reference
            distribution.
        ``r`` : float
            Relative increase in variance due to missing data.
        ``lambda_`` : float
            Fraction of missing information.

    Raises
    ------
    ValueError
        If ``estimates`` and ``variances`` have different lengths, or if
        fewer than 2 imputations are provided.

    Notes
    -----
    The formulas follow van Buuren (2018) §2.3 and Rubin (1987) §3.3::

        q_bar  = (1/m) * sum(q_i)
        u_bar  = (1/m) * sum(u_i)
        b      = 1/(m-1) * sum((q_i - q_bar)^2)
        t      = u_bar + (1 + 1/m) * b
        r      = (1 + 1/m) * b / u_bar
        lambda = (1 + 1/m) * b / t
        df     = (m - 1) * (1 + 1/lambda)^2   (Rubin 1987 / Barnard & Rubin 1999)

    References
    ----------
    * Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
      Wiley. Chapter 3.
    * van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.),
      §2.3. https://stefvanbuuren.name/fimd/

    Examples
    --------
    >>> estimates = [2.1, 2.3, 1.9, 2.0, 2.2]
    >>> variances = [0.04, 0.05, 0.03, 0.04, 0.04]
    >>> result = pool_scalar_estimates(estimates, variances)
    >>> round(result["q_bar"], 1)
    2.1
    >>> result["t"] > result["u_bar"]
    True
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)

    if q.ndim != 1 or u.ndim != 1:
        raise ValueError("estimates and variances must be 1-D sequences.")
    if len(q) != len(u):
        raise ValueError(
            f"estimates (len={len(q)}) and variances (len={len(u)}) must have "
            "the same length."
        )
    m = len(q)
    if m < 2:
        raise ValueError(
            f"At least 2 imputations are required for pooling; got m={m}."
        )
    if np.any(u < 0):
        raise ValueError("All variances must be non-negative.")

    q_bar = q.mean()
    u_bar = u.mean()
    b = np.var(q, ddof=1)          # between-imputation variance
    t = u_bar + (1.0 + 1.0 / m) * b  # total variance

    # Avoid division-by-zero for degenerate cases (perfect pooling)
    if t == 0.0:
        r = 0.0
        lambda_ = 0.0
        df = float("inf")
    else:
        r = (1.0 + 1.0 / m) * b / u_bar if u_bar > 0 else float("inf")
        lambda_ = (1.0 + 1.0 / m) * b / t
        if lambda_ == 0.0:
            df = float("inf")
        else:
            df = (m - 1) * (1.0 / lambda_ + 1.0) ** 2

    return {
        "q_bar": float(q_bar),
        "u_bar": float(u_bar),
        "b": float(b),
        "t": float(t),
        "df": float(df),
        "r": float(r),
        "lambda_": float(lambda_),
    }


def pool_linear_regression_results(
    coefs: np.ndarray,
    covs: np.ndarray,
) -> dict:
    """Pool linear regression results from multiple imputed datasets.

    Applies Rubin's multivariate combining rules to an array of *m*
    coefficient vectors and their associated variance–covariance matrices
    to yield a single pooled coefficient vector and pooled covariance
    matrix.

    Parameters
    ----------
    coefs : np.ndarray, shape (m, p)
        Coefficient vectors from each of the *m* imputed datasets.
        *m* must be >= 2; *p* is the number of model parameters.
    covs : np.ndarray, shape (m, p, p)
        Variance–covariance matrices (one per imputed dataset).
        Each matrix must be symmetric positive semi-definite.

    Returns
    -------
    dict with keys
        ``coef_`` : np.ndarray, shape (p,)
            Pooled (averaged) coefficient vector.
        ``cov_`` : np.ndarray, shape (p, p)
            Pooled total variance–covariance matrix.
        ``se_`` : np.ndarray, shape (p,)
            Pooled standard errors (sqrt of diagonal of ``cov_``).
        ``t_`` : np.ndarray, shape (p,)
            Per-parameter *t*-statistics (``coef_ / se_``).
        ``df_`` : np.ndarray, shape (p,)
            Per-parameter Barnard–Rubin degrees of freedom.

    Raises
    ------
    ValueError
        If array shapes are inconsistent or *m* < 2.

    Notes
    -----
    Multivariate pooling follows van Buuren (2018) §2.3 (applied
    parameter-by-parameter):

    * ``coef_[j]  = mean_i(coefs[i, j])``
    * ``U_bar[j,j] = mean_i(covs[i, j, j])``  (within-imputation)
    * ``B[j,j]    = var_i(coefs[i, j])``       (between-imputation)
    * ``T[j,j]    = U_bar[j,j] + (1 + 1/m)*B[j,j]``
    * Off-diagonals follow the same additive rule applied element-wise.

    References
    ----------
    * Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
      Wiley. Chapter 3.
    * van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.),
      §2.3. https://stefvanbuuren.name/fimd/

    Examples
    --------
    Pool 3 imputations of a simple ``y = beta0 + beta1 * x`` model:

    >>> import numpy as np
    >>> coefs = np.array([[1.0, 2.0], [1.1, 1.9], [0.9, 2.1]])
    >>> covs  = np.array([np.eye(2)*0.01, np.eye(2)*0.01, np.eye(2)*0.01])
    >>> result = pool_linear_regression_results(coefs, covs)
    >>> result["coef_"].round(4)
    array([1., 2.])
    >>> result["se_"].shape
    (2,)
    """
    coefs = np.asarray(coefs, dtype=float)
    covs = np.asarray(covs, dtype=float)

    if coefs.ndim != 2:
        raise ValueError(f"coefs must be 2-D (m, p); got shape {coefs.shape}.")
    if covs.ndim != 3:
        raise ValueError(f"covs must be 3-D (m, p, p); got shape {covs.shape}.")

    m, p = coefs.shape
    if m < 2:
        raise ValueError(
            f"At least 2 imputations are required for pooling; got m={m}."
        )
    if covs.shape != (m, p, p):
        raise ValueError(
            f"covs shape {covs.shape} is inconsistent with coefs shape {coefs.shape}. "
            f"Expected ({m}, {p}, {p})."
        )

    # Pooled coefficient vector: simple mean across imputations
    coef_ = coefs.mean(axis=0)                   # (p,)

    # Within-imputation variance (mean of covariance matrices)
    u_bar = covs.mean(axis=0)                    # (p, p)

    # Between-imputation variance
    diffs = coefs - coef_[np.newaxis, :]         # (m, p)
    b = (diffs[:, :, np.newaxis] * diffs[:, np.newaxis, :]).mean(axis=0) * m / (m - 1)
    # Equivalent to: sum_i outer(qi-qbar, qi-qbar) / (m-1)

    # Total variance–covariance matrix
    cov_ = u_bar + (1.0 + 1.0 / m) * b          # (p, p)

    # Per-parameter quantities
    diag_t = np.diag(cov_)                       # (p,)
    diag_ub = np.diag(u_bar)
    diag_b = np.diag(b)

    se_ = np.sqrt(np.maximum(diag_t, 0.0))       # guard against tiny negatives
    t_ = np.where(se_ > 0, coef_ / se_, np.inf)

    # Barnard–Rubin df per parameter
    lambda_j = np.where(
        diag_t > 0,
        (1.0 + 1.0 / m) * diag_b / diag_t,
        0.0,
    )
    df_ = np.where(
        lambda_j > 0,
        (m - 1) * (1.0 / lambda_j + 1.0) ** 2,
        np.inf,
    )

    return {
        "coef_": coef_,
        "cov_": cov_,
        "se_": se_,
        "t_": t_,
        "df_": df_,
    }
