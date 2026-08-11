"""Multiple Imputation pooling utilities based on Rubin's Rules.

This module provides functions to combine parameter estimates obtained
from *m* independently imputed datasets into a single valid inference,
following the combining rules of:

* Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
  Wiley, New York.
* van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
  CRC/Chapman & Hall.  https://stefvanbuuren.name/fimd/

Public API
----------
pool_scalar_estimates
    Pool m scalar estimates (e.g. a single regression coefficient or a
    group mean) together with their within-imputation variances.
pool_linear_regression_results
    Pool full (m, p) coefficient arrays and (m, p, p) covariance matrices
    from *m* linear-regression fits.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Scalar pooling
# ---------------------------------------------------------------------------

def pool_scalar_estimates(
    estimates: Sequence[float],
    variances: Sequence[float],
) -> Dict[str, float]:
    """Pool *m* scalar estimates using Rubin's combining rules.

    Given *m* independent analyses of *m* imputed datasets this function
    computes the pooled point estimate, its total variance, and several
    diagnostic quantities that characterise the impact of missing data.

    Parameters
    ----------
    estimates : sequence of float, length m
        Scalar point estimates from each imputed dataset (e.g. a
        regression coefficient, a group mean, a proportion).
    variances : sequence of float, length m
        Within-imputation sampling variances corresponding to each
        element of *estimates*.  Must be non-negative and have the
        same length as *estimates*.

    Returns
    -------
    dict with keys:
        ``"q_bar"``
            Pooled point estimate — simple mean of the *m* estimates.
        ``"u_bar"``
            Within-imputation variance — mean of the *m* sampling
            variances.
        ``"b"``
            Between-imputation variance — sample variance of the *m*
            point estimates (uses ``ddof=1``).
        ``"t"``
            Total variance, ``u_bar + (1 + 1/m) * b``.
        ``"df"``
            Approximate degrees of freedom from Rubin's (1987)
            large-sample formula.
        ``"r"``
            Relative increase in variance due to non-response,
            ``(1 + 1/m) * b / u_bar``.
        ``"lambda"``
            Proportion of total variance attributable to missingness,
            ``r / (r + 1)``.
        ``"fmi"``
            Small-sample fraction of missing information,
            ``(r + 2 / (df + 3)) / (r + 1)``.

    Raises
    ------
    ValueError
        If *estimates* and *variances* have different lengths, or if
        fewer than 2 imputations are provided, values are non-finite,
        or a within-imputation variance is negative.

    Notes
    -----
    The combining rules implemented here follow equations (3.1)–(3.6)
    of Rubin (1987) and the accessible summary in Section 2.3 of
    van Buuren (2018).

    The degrees-of-freedom formula is:

    .. math::

        \\nu = (m - 1)\\left(1 + \\frac{\\bar{U}}{(1 + m^{-1})B}\\right)^2

    which is Rubin's (1987) large-sample formula.  When ``b == 0``
    (all imputed estimates are identical, meaning no between-imputation
    variability) ``df`` is set to ``inf`` and ``r`` / ``lambda`` /
    ``fmi`` to 0.
    When ``u_bar == 0`` but ``b > 0``, ``r`` is ``inf`` and ``df`` is
    set to ``(m - 1)``.

    Examples
    --------
    Three imputations of a group mean, each with the same estimated
    standard error:

    >>> estimates = [1.0, 3.0, 5.0]
    >>> variances = [0.25, 0.25, 0.25]
    >>> result = pool_scalar_estimates(estimates, variances)
    >>> result["q_bar"]
    3.0
    >>> result["u_bar"]
    0.25
    >>> result["b"] > 0
    True
    >>> result["t"] > result["u_bar"]
    True

    Identical estimates (b == 0) — no between-imputation variance:

    >>> result2 = pool_scalar_estimates([2.0, 2.0, 2.0], [0.1, 0.1, 0.1])
    >>> result2["b"]
    0.0
    >>> result2["df"]
    inf
    >>> result2["r"]
    0.0
    """
    estimates = list(estimates)
    variances = list(variances)

    if len(estimates) != len(variances):
        raise ValueError(
            f"estimates (length {len(estimates)}) and variances "
            f"(length {len(variances)}) must have the same length."
        )
    m = len(estimates)
    if m < 2:
        raise ValueError(
            f"At least 2 imputations are required for pooling; got {m}."
        )

    q_arr = np.array(estimates, dtype=float)
    u_arr = np.array(variances, dtype=float)

    if not np.isfinite(q_arr).all():
        raise ValueError("estimates must contain only finite values.")
    if not np.isfinite(u_arr).all():
        raise ValueError("variances must contain only finite values.")
    if (u_arr < 0).any():
        raise ValueError("variances must be non-negative.")

    q_bar = float(q_arr.mean())                          # pooled estimate
    u_bar = float(u_arr.mean())                          # within-imputation variance
    b = float(np.var(q_arr, ddof=1))                    # between-imputation variance
    t = u_bar + (1.0 + 1.0 / m) * b                    # total variance

    # --- handle degenerate cases before computing r and df ---
    if b == 0.0:
        # All estimates are identical: no between-imputation variability.
        # r = 0, df = inf, lambda = fmi = 0.
        r = 0.0
        df = float("inf")
        lam = 0.0
        fmi = 0.0
    elif u_bar == 0.0:
        # No within-imputation variance but estimates differ.
        # r = inf, df degenerates to (m-1).
        r = float("inf")
        df = float(m - 1)
        lam = 1.0
        fmi = 1.0
    else:
        r = (1.0 + 1.0 / m) * b / u_bar
        # Rubin (1987) degrees-of-freedom formula
        df = (m - 1) * (1.0 + u_bar / ((1.0 + 1.0 / m) * b)) ** 2
        lam = r / (r + 1.0)
        fmi = (r + 2.0 / (df + 3.0)) / (r + 1.0)

    return {
        "q_bar": q_bar,
        "u_bar": u_bar,
        "b": b,
        "t": t,
        "df": df,
        "r": r,
        "lambda": lam,
        "fmi": fmi,
    }


# ---------------------------------------------------------------------------
# Regression pooling
# ---------------------------------------------------------------------------

def pool_linear_regression_results(
    coefs: np.ndarray,
    covs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pool *m* linear-regression results using Rubin's combining rules.

    Applies the multivariate version of Rubin's Rules to a collection of
    coefficient vectors and covariance matrices obtained by fitting the
    same regression model on *m* imputed datasets.

    Parameters
    ----------
    coefs : np.ndarray of shape (m, p)
        Regression coefficient vectors from *m* imputed datasets.
        Each row corresponds to one imputed dataset.
    covs : np.ndarray of shape (m, p, p)
        Sampling covariance matrices from *m* imputed datasets.
        Each ``covs[i]`` is the estimated parameter covariance matrix
        for dataset *i* (e.g. the output of ``np.linalg.inv(X.T @ X) *
        sigma_sq``).

    Returns
    -------
    dict with keys:
        ``"coef_"`` : np.ndarray of shape (p,)
            Pooled coefficient vector — element-wise mean of the *m*
            coefficient vectors.
        ``"cov_"`` : np.ndarray of shape (p, p)
            Pooled covariance matrix using the multivariate Rubin
            formula:  ``U_bar + (1 + 1/m) * B``, where *U_bar* is the
            mean of the *m* covariance matrices and *B* is the
            between-imputation covariance of the coefficient vectors.
        ``"se_"`` : np.ndarray of shape (p,)
            Standard errors — square root of the diagonal of ``"cov_"``.
        ``"t_"`` : np.ndarray of shape (p,)
            Total variance for each parameter — the diagonal of
            ``"cov_"``.
        ``"df_"`` : np.ndarray of shape (p,)
            Approximate per-parameter degrees of freedom using the
            scalar Rubin formula applied to each diagonal element.

    Raises
    ------
    ValueError
        If *coefs* and *covs* have incompatible shapes or fewer than
        2 imputations are provided, or if coefficients/covariances are
        non-finite, non-symmetric, or the sampling covariance is not
        positive semidefinite.

    Notes
    -----
    The multivariate combining rules follow van Buuren (2018) Section 2.4
    and the original development in Rubin (1987, Chapter 3).

    Between-imputation covariance:

    .. math::

        B = \\frac{1}{m-1} \\sum_{i=1}^{m}
            (\\hat{\\beta}_i - \\bar{\\beta})
            (\\hat{\\beta}_i - \\bar{\\beta})^\\top

    Total covariance:

    .. math::

        T = \\bar{U} + \\left(1 + \\frac{1}{m}\\right) B

    Examples
    --------
    Synthetic example with *m* = 2 imputations and *p* = 2 parameters:

    >>> import numpy as np
    >>> coefs = np.array([[1.0, 2.0], [1.1, 1.9]])
    >>> cov0 = np.array([[0.04, 0.01], [0.01, 0.04]])
    >>> covs = np.array([cov0, cov0])
    >>> result = pool_linear_regression_results(coefs, covs)
    >>> result["coef_"]
    array([1.05, 1.95])
    >>> result["se_"].shape
    (2,)
    >>> bool(np.all(result["se_"] > 0))
    True
    """
    coefs = np.asarray(coefs, dtype=float)
    covs = np.asarray(covs, dtype=float)

    if coefs.ndim != 2:
        raise ValueError(
            f"coefs must be a 2-D array of shape (m, p); got shape {coefs.shape}."
        )
    if covs.ndim != 3:
        raise ValueError(
            f"covs must be a 3-D array of shape (m, p, p); got shape {covs.shape}."
        )

    m, p = coefs.shape
    if covs.shape != (m, p, p):
        raise ValueError(
            f"covs shape {covs.shape} is inconsistent with coefs shape {coefs.shape}; "
            f"expected ({m}, {p}, {p})."
        )
    if m < 2:
        raise ValueError(
            f"At least 2 imputations are required for pooling; got {m}."
        )
    if not np.isfinite(coefs).all():
        raise ValueError("coefs must contain only finite values.")
    if not np.isfinite(covs).all():
        raise ValueError("covs must contain only finite values.")
    if not np.allclose(covs, np.swapaxes(covs, 1, 2), rtol=1e-10, atol=1e-12):
        raise ValueError("Each covs matrix must be symmetric.")

    covariance_scale = np.maximum(np.max(np.abs(covs), axis=(1, 2)), 1.0)
    eigenvalue_tolerance = np.finfo(float).eps * 100 * covariance_scale
    if np.any(np.linalg.eigvalsh(covs) < -eigenvalue_tolerance[:, None]):
        raise ValueError("Each covs matrix must be positive semidefinite.")

    # Pooled coefficient vector
    coef_bar = coefs.mean(axis=0)                        # shape (p,)

    # Within-imputation covariance: mean of the m matrices
    u_bar = covs.mean(axis=0)                            # shape (p, p)

    # Between-imputation covariance
    diffs = coefs - coef_bar                             # shape (m, p)
    b_mat = (diffs[:, :, None] * diffs[:, None, :]).mean(axis=0) * m / (m - 1)  # unbiased

    # Total covariance
    t_mat = u_bar + (1.0 + 1.0 / m) * b_mat             # shape (p, p)

    # Per-parameter total variance, SE, and df
    t_diag = np.diag(t_mat)                              # shape (p,)
    se = np.sqrt(np.maximum(t_diag, 0.0))               # guard against tiny negatives
    u_diag = np.diag(u_bar)
    b_diag = np.diag(b_mat)

    # Degrees of freedom (scalar Rubin formula applied element-wise)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_arr = np.where(
            b_diag > 0,
            (m - 1) * (1.0 + u_diag / ((1.0 + 1.0 / m) * b_diag)) ** 2,
            np.inf,
        )

    return {
        "coef_": coef_bar,
        "cov_": t_mat,
        "se_": se,
        "t_": t_diag,
        "df_": df_arr,
    }
