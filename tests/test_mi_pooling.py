"""Tests for missingly.mi — Rubin's Rules pooling utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.mi import pool_scalar_estimates, pool_linear_regression_results


# ---------------------------------------------------------------------------
# Test 1 — pool_scalar_estimates: analytic check
# ---------------------------------------------------------------------------

class TestPoolScalarEstimates:
    def test_basic_analytic(self):
        """m=2, symmetric estimates → q_bar exact, b>0, t>u_bar."""
        result = pool_scalar_estimates([1.0, 3.0], [0.25, 0.25])

        assert result["q_bar"] == pytest.approx(2.0)
        assert result["u_bar"] == pytest.approx(0.25)
        assert result["b"] > 0
        assert result["t"] > result["u_bar"]

    def test_five_estimates(self):
        """Five estimates with known mean."""
        ests = [1.0, 2.0, 3.0, 4.0, 5.0]
        vars_ = [0.1] * 5
        result = pool_scalar_estimates(ests, vars_)
        assert result["q_bar"] == pytest.approx(3.0)
        assert result["u_bar"] == pytest.approx(0.1)

    def test_lambda_between_zero_and_one(self):
        result = pool_scalar_estimates([1.0, 1.5, 2.0], [0.1, 0.1, 0.1])
        assert 0.0 <= result["lambda_"] <= 1.0

    def test_identical_estimates_zero_between_variance(self):
        result = pool_scalar_estimates([2.0, 2.0, 2.0], [0.1, 0.1, 0.1])
        assert result["b"] == pytest.approx(0.0, abs=1e-12)
        assert result["q_bar"] == pytest.approx(2.0)

    def test_raises_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            pool_scalar_estimates([1.0, 2.0], [0.1])

    def test_raises_too_few_imputations(self):
        with pytest.raises(ValueError, match="At least 2"):
            pool_scalar_estimates([1.0], [0.1])

    def test_raises_negative_variance(self):
        with pytest.raises(ValueError, match="non-negative"):
            pool_scalar_estimates([1.0, 2.0], [-0.1, 0.1])


# ---------------------------------------------------------------------------
# Test 2 — pool_linear_regression_results: shape, PD, bounds
# ---------------------------------------------------------------------------

class TestPoolLinearRegressionResults:
    def _make_inputs(self, m=3, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        true_coefs = np.array([1.0, 2.0])
        coefs = true_coefs + rng.normal(0, noise, size=(m, 2))
        covs = np.stack([np.eye(2) * 0.01] * m)
        return coefs, covs, true_coefs

    def test_coef_within_bounds(self):
        coefs, covs, true_coefs = self._make_inputs()
        result = pool_linear_regression_results(coefs, covs)
        pooled = result["coef_"]
        assert np.all(pooled >= coefs.min(axis=0) - 1e-10)
        assert np.all(pooled <= coefs.max(axis=0) + 1e-10)

    def test_cov_positive_definite(self):
        coefs, covs, _ = self._make_inputs()
        result = pool_linear_regression_results(coefs, covs)
        eigvals = np.linalg.eigvalsh(result["cov_"])
        assert np.all(eigvals > 0), f"Eigenvalues not all positive: {eigvals}"

    def test_output_shapes(self):
        coefs, covs, _ = self._make_inputs(m=5)
        result = pool_linear_regression_results(coefs, covs)
        assert result["coef_"].shape == (2,)
        assert result["cov_"].shape == (2, 2)
        assert result["se_"].shape == (2,)
        assert result["t_"].shape == (2,)
        assert result["df_"].shape == (2,)

    def test_raises_wrong_coef_dims(self):
        with pytest.raises(ValueError, match="2-D"):
            pool_linear_regression_results(np.ones(3), np.ones((3, 2, 2)))

    def test_raises_shape_mismatch(self):
        with pytest.raises(ValueError, match="inconsistent"):
            pool_linear_regression_results(np.ones((3, 2)), np.ones((3, 3, 3)))


# ---------------------------------------------------------------------------
# Test 3 — end-to-end: impute_mice + pool_scalar_estimates
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Smoke test: MICE imputation followed by OLS pooling."""

    @pytest.fixture()
    def simple_df(self):
        rng = np.random.default_rng(42)
        n = 60
        x = rng.uniform(0, 10, n)
        true_beta1 = 3.0
        y = 1.0 + true_beta1 * x + rng.normal(0, 1, n)
        mask = rng.random(n) < 0.3
        y_missing = y.copy().astype(float)
        y_missing[mask] = np.nan
        return pd.DataFrame({"x": x, "y": y_missing}), true_beta1

    def test_pooled_beta_near_truth(self, simple_df):
        """Pooled beta1 should fall within ±0.5 of the true value (3.0)."""
        pytest.importorskip("sklearn")
        from missingly import impute_mice
        from sklearn.linear_model import LinearRegression

        df, true_beta1 = simple_df
        m = 5
        imputed_dfs = impute_mice(df, n_imputations=m, random_state=0)

        beta1_ests, beta1_vars = [], []
        for df_i in imputed_dfs:
            X = df_i[["x"]].values
            y = df_i["y"].values
            reg = LinearRegression().fit(X, y)
            beta1_ests.append(float(reg.coef_[0]))
            # Approximate sampling variance via residual variance / SS_x
            y_hat = reg.predict(X)
            resid_var = float(np.var(y - y_hat, ddof=2))
            ss_x = float(np.sum((X - X.mean()) ** 2))
            beta1_vars.append(resid_var / ss_x if ss_x > 0 else 1e-6)

        result = pool_scalar_estimates(beta1_ests, beta1_vars)
        pooled_beta1 = result["q_bar"]
        assert abs(pooled_beta1 - true_beta1) < 0.5, (
            f"Pooled beta1={pooled_beta1:.4f} too far from true={true_beta1}"
        )
