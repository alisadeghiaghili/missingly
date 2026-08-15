"""Behavioral contracts for strict-mode configuration resolution."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import ConfigurationError
from missingly.impute import impute_gb, impute_logreg, impute_rf


@pytest.fixture()
def binary_frame() -> pd.DataFrame:
    """Return a binary target with one eligible row for imputation."""
    return pd.DataFrame(
        {
            "predictor": [0.0, 1.0, 0.0, 1.0],
            "outcome": pd.Categorical(["no", "yes", "no", pd.NA]),
        }
    )


class _NeverRunEstimator:
    """Record accidental estimator execution after an invalid configuration."""

    fit_calls = 0
    predict_calls = 0

    def __init__(self, **kwargs: object) -> None:
        """Accept the production estimator constructor keywords."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NeverRunEstimator":
        """Fail loudly if validation was not completed before fitting."""
        type(self).fit_calls += 1
        raise AssertionError("an invalid strict_mode must prevent estimator fitting")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fail loudly if validation was not completed before prediction."""
        type(self).predict_calls += 1
        raise AssertionError("an invalid strict_mode must prevent prediction")


def _reset_never_run_estimator() -> None:
    """Clear class-level execution counters between independent assertions."""
    _NeverRunEstimator.fit_calls = 0
    _NeverRunEstimator.predict_calls = 0


def test_logreg_rejects_non_boolean_override_before_model_execution(
    monkeypatch,
    binary_frame,
):
    """A truthy string is not a valid strict-mode override or estimator input."""
    _reset_never_run_estimator()
    monkeypatch.setattr("missingly.impute.LogisticRegression", _NeverRunEstimator)
    original = binary_frame.copy(deep=True)

    with pytest.raises(ConfigurationError, match="strict_mode"):
        impute_logreg(binary_frame, strict_mode="yes")

    assert _NeverRunEstimator.fit_calls == 0
    assert _NeverRunEstimator.predict_calls == 0
    pd.testing.assert_frame_equal(binary_frame, original)


@pytest.mark.parametrize(
    ("imputer", "regressor_name", "classifier_name"),
    [
        (impute_rf, "RandomForestRegressor", "RandomForestClassifier"),
        (impute_gb, "GradientBoostingRegressor", "GradientBoostingClassifier"),
    ],
)
def test_model_imputers_reject_integer_strict_mode_before_shared_estimator_execution(
    monkeypatch,
    binary_frame,
    imputer: Callable[..., pd.DataFrame],
    regressor_name: str,
    classifier_name: str,
):
    """The shared RF/GB path rejects ``1`` instead of treating it as ``True``."""
    _reset_never_run_estimator()
    monkeypatch.setattr(f"missingly.impute.{regressor_name}", _NeverRunEstimator)
    monkeypatch.setattr(f"missingly.impute.{classifier_name}", _NeverRunEstimator)
    original = binary_frame.copy(deep=True)

    with pytest.raises(ConfigurationError, match="strict_mode"):
        imputer(binary_frame, strict_mode=1)

    assert _NeverRunEstimator.fit_calls == 0
    assert _NeverRunEstimator.predict_calls == 0
    pd.testing.assert_frame_equal(binary_frame, original)


def test_invalid_global_strict_mode_fails_before_logreg_model_execution(
    monkeypatch,
    binary_frame,
):
    """An invalid package default must fail before it can affect model behavior."""
    _reset_never_run_estimator()
    monkeypatch.setattr("missingly.impute.LogisticRegression", _NeverRunEstimator)
    monkeypatch.setattr("missingly.impute._config.strict_mode", "enabled")
    original = binary_frame.copy(deep=True)

    with pytest.raises(ConfigurationError, match="missingly.config.strict_mode"):
        impute_logreg(binary_frame)

    assert _NeverRunEstimator.fit_calls == 0
    assert _NeverRunEstimator.predict_calls == 0
    pd.testing.assert_frame_equal(binary_frame, original)


def test_boolean_override_does_not_consult_an_invalid_global_default(
    monkeypatch,
    binary_frame,
):
    """A valid explicit override takes precedence over an invalid global setting."""
    monkeypatch.setattr("missingly.impute._config.strict_mode", "enabled")

    result = impute_logreg(binary_frame, strict_mode=False)

    assert not result["outcome"].isna().any()
    pd.testing.assert_series_equal(
        result.loc[:2, "outcome"],
        binary_frame.loc[:2, "outcome"],
    )
