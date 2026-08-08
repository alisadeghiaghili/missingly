"""Tests for missingly.mi pooling utilities (Rubin's Rules).

Includes:
  - Unit tests for pool_scalar_estimates
  - Unit tests for pool_linear_regression_results
  - Input validation smoke tests
  - Integration test: full MI workflow with impute_mice + pool_scalar_estimates
"""

from __future__ import annotations

import numpy as np
import pytest

from missingly.mi import pool_scalar_estimates, pool_linear_regression_results


# ---------------------------------------------------------------------------
# Test 1: pool_scalar_estimates — basic correctness
# ---------------------------------------------------------------------------

class TestPoolScalarEstimates:
    estimates = [1.0, 3.0, 5.0]
    variances = [0.25, 0.25, 0.25]

    def test_q_bar(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert result["q_bar"] == pytest.approx(3.0)

    def test_u_bar(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert result["u_bar"] == pytest.approx(0.25)

    def test_b_positive(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert result["b"] > 0

    def test_total_variance_gt_within(self):
        """Total variance must exceed within-imputation variance."""
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert result["t"] > result["u_bar"]

    def test_lambda_in_unit_interval(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert 0.0 <= result["lambda"] <= 1.0

    def test_fmi_in_unit_interval(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert 0.0 <= result["fmi"] <= 1.0

    def test_df_positive(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        assert result["df"] > 0

    def test_all_keys_present(self):
        result = pool_scalar_estimates(self.estimates, self.variances)
        expected_keys = {"q_bar", "u_bar", "b", "t", "df", "r", "lambda", "fmi"}
        assert set(result.keys()) == expected_keys

    def test_identical_estimates_b_zero(self):
        """When all estimates are identical b should be 0."""
        result = pool_scalar_estimates([2.0, 2.0, 2.0], [0.1, 0.1, 0.1])
        assert result["b"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 2: pool_linear_regression_results — basic correctness
# ---------------------------------------------------------------------------

class TestPoolLinearRegressionResults:
    m, p = 3, 2
    rng = np.random.default_rng(42)
    coefs = np.array([[1.0, 2.0], [0.95, 2.05], [1.05, 1.95]])
    base_cov = np.array([[0.04, 0.005], [0.005, 0.04]])
    covs = np.stack([base_cov] * m)

    def test_coef_shape(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        assert result["coef_"].shape == (self.p,)

    def test_coef_values(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        assert result["coef_"] == pytest.approx([1.0, 2.0], abs=0.1)

    def test_cov_shape(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        assert result["cov_"].shape == (self.p, self.p)

    def test_cov_positive_definite(self):
        """Pooled covariance matrix must be positive-definite."""
        result = pool_linear_regression_results(self.coefs, self.covs)
        eigenvalues = np.linalg.eigvalsh(result["cov_"])
        assert np.all(eigenvalues > 0), f"Non-positive eigenvalues: {eigenvalues}"

    def test_se_shape_and_positive(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        assert result["se_"].shape == (self.p,)
        assert np.all(result["se_"] > 0)

    def test_df_shape_and_positive(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        assert result["df_"].shape == (self.p,)
        assert np.all(result["df_"] > 0)

    def test_all_keys_present(self):
        result = pool_linear_regression_results(self.coefs, self.covs)
        expected_keys = {"coef_", "cov_", "se_", "t_", "df_"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3: ValueError on mismatched / invalid inputs
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            pool_scalar_estimates([1.0, 2.0], [0.1])

    def test_single_imputation_raises_scalar(self):
        with pytest.raises(ValueError, match="At least 2"):
            pool_scalar_estimates([1.0], [0.1])

    def test_single_imputation_raises_regression(self):
        with pytest.raises(ValueError, match="At least 2"):
            pool_linear_regression_results(
                np.array([[1.0, 2.0]]),
                np.array([[[0.04, 0.0], [0.0, 0.04]]]),
            )

    def test_wrong_coefs_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D array"):
            pool_linear_regression_results(
                np.array([1.0, 2.0]),
                np.zeros((2, 2, 2)),
            )

    def test_incompatible_covs_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            pool_linear_regression_results(
                np.ones((3, 2)),
                np.ones((3, 3, 3)),
            )


# ---------------------------------------------------------------------------
# Test 4 (integration): full MI workflow with impute_mice
# ---------------------------------------------------------------------------

class TestMIIntegration:
    """End-to-end test: impute_mice → LinearRegression → pool_scalar_estimates.

    True relationship: y = 2 * x + noise.
    We expect the pooled beta1 (slope) to be close to 2.0.
    """

    def test_pooled_beta1_close_to_truth(self):
        """Pooled slope from 5 MI chains should be in (1.7, 2.3)."""
        from sklearn.linear_model import LinearRegression
        from missingly.impute import impute_mice

        rng = np.random.default_rng(0)
        n = 100
        x = rng.uniform(0, 10, n)
        y = 2.0 * x + rng.normal(0, 1, n)

        # MCAR: remove ~30% of y
        mask = rng.random(n) < 0.30
        y_missing = y.copy()
        y_missing[mask] = np.nan

        import pandas as pd
        df = pd.DataFrame({"x": x, "y": y_missing})

        m = 5
        dfs = impute_mice(df, n_imputations=m, random_state=42)

        beta1_ests, beta1_vars = [], []
        for d in dfs:
            reg = LinearRegression().fit(d[["x"]], d["y"])
            beta1 = float(reg.coef_[0])
            resid = d["y"] - reg.predict(d[["x"]])
            ss_x = float(((d["x"] - d["x"].mean()) ** 2).sum())
            # Variance of slope estimate: sigma^2 / SS_x
            sigma2 = float(np.var(resid, ddof=2))
            var_beta1 = sigma2 / ss_x if ss_x > 0 else 1e-6
            beta1_ests.append(beta1)
            beta1_vars.append(max(var_beta1, 1e-10))  # guard numerical zeros

        result = pool_scalar_estimates(beta1_ests, beta1_vars)

        assert 1.7 < result["q_bar"] < 2.3, (
            f"Pooled beta1 = {result['q_bar']:.4f} is outside (1.7, 2.3)"
        )
        assert result["t"] > 0, "Total variance must be positive"
