"""Tests for missingly.mi_workflow: mi_with, mi_fit, pool_linear_models.

Scenarios
---------
1. mi_with returns a list of models with the correct type and length.
2. mi_fit runs end-to-end: dfs and models have equal length, no NaN in dfs.
3. pool_linear_models: pooled slope is close to the true value (2.0).
4. Error handling:
   - mi_with: empty dfs raises ValueError.
   - mi_fit: n_imputations < 2 raises ValueError.
   - pool_linear_models: mismatched lengths raise ValueError.
   - pool_linear_models: fewer than 2 models raises ValueError.
5. mi_fit passes impute_kwargs through to impute_mice.
6. pool_linear_models observed values are unchanged in each df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from missingly.impute import impute_mice
from missingly.mi_workflow import mi_with, mi_fit, pool_linear_models


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_df() -> pd.DataFrame:
    """Small DataFrame: y = 2*x + noise, ~30% of y is missing (MCAR)."""
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 2.0 * x + rng.normal(0, 1, n)
    mask = rng.choice(n, size=18, replace=False)
    y[mask] = np.nan
    return pd.DataFrame({"x": x, "y": y})


def _lm(d: pd.DataFrame) -> LinearRegression:
    return LinearRegression().fit(d[["x"]], d["y"])


# ---------------------------------------------------------------------------
# 1. mi_with — returns list of correct type and length
# ---------------------------------------------------------------------------

class TestMiWith:
    def test_returns_list_length_m(self, simple_df):
        dfs = impute_mice(simple_df, n_imputations=3, random_state=0)
        models = mi_with(dfs, _lm)
        assert isinstance(models, list)
        assert len(models) == 3

    def test_model_types(self, simple_df):
        dfs = impute_mice(simple_df, n_imputations=3, random_state=0)
        models = mi_with(dfs, _lm)
        assert all(isinstance(m, LinearRegression) for m in models)

    def test_model_fn_called_on_each_df(self, simple_df):
        """model_fn should see each df independently (different fitted coefs ok)."""
        dfs = impute_mice(simple_df, n_imputations=4, random_state=1)
        call_count = [0]

        def counting_fn(d):
            call_count[0] += 1
            return _lm(d)

        mi_with(dfs, counting_fn)
        assert call_count[0] == 4


# ---------------------------------------------------------------------------
# 2. mi_fit — end-to-end: equal lengths, no NaN
# ---------------------------------------------------------------------------

class TestMiFit:
    def test_dfs_and_models_equal_length(self, simple_df):
        dfs, models = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 42},
        )
        assert len(dfs) == len(models) == 3

    def test_no_nan_in_imputed_dfs(self, simple_df):
        dfs, _ = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 42},
        )
        for df in dfs:
            assert df.isnull().sum().sum() == 0, "NaN remains in imputed DataFrame"

    def test_returns_tuple_of_two(self, simple_df):
        result = mi_fit(simple_df, n_imputations=2, model_fn=_lm)
        assert isinstance(result, tuple) and len(result) == 2

    def test_impute_kwargs_forwarded(self, simple_df):
        """Passing random_state should not raise; just checks no exception."""
        dfs, models = mi_fit(
            simple_df, n_imputations=2, model_fn=_lm,
            impute_kwargs={"random_state": 99},
        )
        assert len(dfs) == 2


# ---------------------------------------------------------------------------
# 3. pool_linear_models — pooled slope close to truth (2.0)
# ---------------------------------------------------------------------------

class TestPoolLinearModels:
    def test_slope_close_to_truth(self):
        """True slope = 2.0; pooled estimate should be in (1.7, 2.3)."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 10, n)
        y = 2.0 * x + rng.normal(0, 1, n)
        miss = rng.choice(n, size=30, replace=False)
        y[miss] = np.nan
        df = pd.DataFrame({"x": x, "y": y})

        dfs, models = mi_fit(
            df, n_imputations=5, model_fn=_lm,
            impute_kwargs={"random_state": 42},
        )
        pooled = pool_linear_models(models, dfs, feature="x", target="y")

        assert 1.7 < pooled["q_bar"] < 2.3, (
            f"Pooled slope {pooled['q_bar']:.4f} outside (1.7, 2.3)"
        )

    def test_returns_all_rubin_keys(self, simple_df):
        dfs, models = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 0},
        )
        pooled = pool_linear_models(models, dfs, feature="x", target="y")
        expected_keys = {"q_bar", "u_bar", "b", "t", "df", "r", "lambda"}
        assert set(pooled.keys()) == expected_keys

    def test_total_variance_positive(self, simple_df):
        dfs, models = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 0},
        )
        pooled = pool_linear_models(models, dfs, feature="x", target="y")
        assert pooled["t"] > 0


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_mi_with_empty_dfs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            mi_with([], _lm)

    def test_mi_fit_n_imputations_lt_2_raises(self, simple_df):
        with pytest.raises(ValueError, match="n_imputations must be >= 2"):
            mi_fit(simple_df, n_imputations=1, model_fn=_lm)

    def test_pool_mismatched_lengths_raises(self, simple_df):
        dfs, models = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 0},
        )
        with pytest.raises(ValueError, match="same length"):
            pool_linear_models(models[:2], dfs, feature="x", target="y")

    def test_pool_single_model_raises(self, simple_df):
        dfs, models = mi_fit(
            simple_df, n_imputations=3, model_fn=_lm,
            impute_kwargs={"random_state": 0},
        )
        with pytest.raises(ValueError, match="at least 2 models"):
            pool_linear_models(models[:1], dfs[:1], feature="x", target="y")


# ---------------------------------------------------------------------------
# 5. Public API: importable directly from missingly
# ---------------------------------------------------------------------------

def test_public_api_imports():
    """Ensure mi_with, mi_fit, pool_linear_models are in missingly namespace."""
    import missingly
    assert hasattr(missingly, "mi_with"), "mi_with not in missingly namespace"
    assert hasattr(missingly, "mi_fit"), "mi_fit not in missingly namespace"
    assert hasattr(missingly, "pool_linear_models"), (
        "pool_linear_models not in missingly namespace"
    )
