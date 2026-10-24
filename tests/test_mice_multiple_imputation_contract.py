"""Multiple imputation needs between-imputation variance.

An estimator that returns point predictions only (``LinearRegression``, a
random forest, ...) makes every chain identical, so Rubin's rules would see zero
between-imputation variance (``df=inf``, ``fmi=0``) and report intervals that
are too narrow. ``impute_mice`` refuses ``n_imputations > 1`` for such an
estimator and draws from the predictive distribution of one that offers
``predict(..., return_std=True)``.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression

from missingly.impute import impute_mice
from missingly.mi import pool_scalar_estimates


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(size=n)})
    frame.loc[rng.choice(n, 30, replace=False), "y"] = np.nan
    return frame


def _imputed_values(frames: List[pd.DataFrame], frame: pd.DataFrame) -> List[np.ndarray]:
    missing = frame["y"].isna().to_numpy()
    return [chain.loc[missing, "y"].to_numpy() for chain in frames]


@pytest.mark.parametrize(
    ("estimator", "name"),
    [
        (LinearRegression(), "LinearRegression"),
        (RandomForestRegressor(n_estimators=3, random_state=0), "RandomForestRegressor"),
    ],
    ids=["linear", "forest"],
)
def test_point_prediction_estimator_cannot_make_several_imputations(
    estimator: object, name: str
) -> None:
    with pytest.raises(ValueError) as caught:
        impute_mice(_frame(), estimator=estimator, n_imputations=4, random_state=0)

    message = str(caught.value)
    assert name in message
    assert "between-imputation variance" in message
    assert "return_std" in message
    assert "n_imputations=1" in message


@pytest.mark.parametrize("option", [{"return_history": True}, {"return_result": True}])
def test_refusal_applies_to_every_return_shape(option: dict) -> None:
    with pytest.raises(ValueError, match="between-imputation variance"):
        impute_mice(
            _frame(), estimator=LinearRegression(), n_imputations=2, max_iter=2, **option
        )


def test_point_prediction_estimator_is_still_fine_for_one_imputation() -> None:
    result = impute_mice(_frame(), estimator=LinearRegression(), n_imputations=1, max_iter=3)
    assert not result.isna().any().any()


@pytest.mark.parametrize(
    "estimator",
    [None, BayesianRidge(), ARDRegression()],
    ids=["default", "bayesian-ridge", "ard"],
)
def test_posterior_sampling_estimator_gives_distinct_data_sets(estimator: object) -> None:
    frame = _frame()
    m = 4

    chains = impute_mice(frame, estimator=estimator, n_imputations=m, max_iter=4, random_state=0)

    assert len(chains) == m
    for left, right in itertools.combinations(_imputed_values(chains, frame), 2):
        assert not np.allclose(left, right)
    for chain in chains:
        pd.testing.assert_series_equal(chain["x"], frame["x"])
        observed = frame["y"].notna()
        pd.testing.assert_series_equal(chain.loc[observed, "y"], frame.loc[observed, "y"])


def test_between_imputation_variance_is_positive_so_rubin_df_is_finite() -> None:
    frame = _frame()
    chains = impute_mice(frame, n_imputations=8, max_iter=4, random_state=0)
    estimates = [float(chain["y"].mean()) for chain in chains]
    variances = [float(chain["y"].var(ddof=1) / len(chain)) for chain in chains]

    pooled = pool_scalar_estimates(estimates, variances)

    assert pooled["b"] > 0
    assert np.isfinite(pooled["df"])
    assert pooled["fmi"] > 0


def test_same_seed_reproduces_posterior_draws() -> None:
    frame = _frame()
    first = impute_mice(frame, estimator=BayesianRidge(), n_imputations=3, random_state=11)
    again = impute_mice(frame, estimator=BayesianRidge(), n_imputations=3, random_state=11)
    for left, right in zip(first, again):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_single_imputation_is_the_first_chain_of_a_multiple_imputation() -> None:
    frame = _frame()
    one = impute_mice(frame, estimator=BayesianRidge(), n_imputations=1, random_state=6)
    three = impute_mice(frame, estimator=BayesianRidge(), n_imputations=3, random_state=6)
    pd.testing.assert_frame_equal(one, three[0], check_exact=True)


def test_typed_result_records_how_values_were_imputed() -> None:
    frame = _frame()
    draws = impute_mice(frame, n_imputations=2, max_iter=2, return_result=True)
    points = impute_mice(
        frame, estimator=LinearRegression(), n_imputations=1, max_iter=2, return_result=True
    )
    assert draws.data.plan.provenance["imputation_values"] == "posterior_predictive_draws"
    assert points.data.plan.provenance["imputation_values"] == "point_predictions"
