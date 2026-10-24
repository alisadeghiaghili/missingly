"""A failing conditional model in ``impute_mice`` is reported, never hidden.

The chained-equations loop used to replace a column whose model raised during
``fit`` or ``predict`` by the observed mean, without a word. It now follows the
same ``strict_mode`` contract as the other model-based imputers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from missingly.exceptions import ImputationError
from missingly.impute import impute_mice


class _FailingRegressor(BaseEstimator, RegressorMixin):
    """Regressor whose fit always raises, to force the failure path."""

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        raise ValueError("forced estimator failure")

    def predict(self, X, return_std=False):  # type: ignore[no-untyped-def]
        zeros = np.zeros(len(X))
        return (zeros, zeros) if return_std else zeros


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    frame = pd.DataFrame({"a": x, "b": 2 * x + rng.normal(scale=0.1, size=40)})
    frame.loc[2::8, "a"] = np.nan
    frame.loc[4::9, "b"] = np.nan
    return frame


def test_failed_fit_raises_by_default_and_names_column_and_cause() -> None:
    with pytest.raises(ImputationError) as caught:
        impute_mice(_frame(), estimator=_FailingRegressor(), max_iter=2, random_state=0)

    assert caught.value.column == "a"
    assert caught.value.strategy == "mice"
    assert "forced estimator failure" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def test_failed_fit_warns_per_column_and_fills_with_the_mean_when_not_strict() -> None:
    frame = _frame()

    with pytest.warns(UserWarning) as record:
        result = impute_mice(
            frame,
            estimator=_FailingRegressor(),
            max_iter=2,
            random_state=0,
            strict_mode=False,
        )

    messages = [str(item.message) for item in record if item.category is UserWarning]
    assert len(messages) == 2
    assert any("'a'" in message for message in messages)
    assert any("'b'" in message for message in messages)
    for column in ("a", "b"):
        missing = frame[column].isna()
        assert np.allclose(result.loc[missing, column], frame[column].mean())


def test_fallback_is_reported_once_per_column_across_chains() -> None:
    with pytest.warns(UserWarning) as record:
        impute_mice(
            _frame(),
            estimator=_FailingRegressor(),
            max_iter=3,
            n_imputations=3,
            random_state=0,
            strict_mode=False,
        )
    assert len([r for r in record if r.category is UserWarning]) == 2


def test_typed_result_records_the_fallback_columns() -> None:
    with pytest.warns(UserWarning):
        result = impute_mice(
            _frame(),
            estimator=_FailingRegressor(),
            max_iter=2,
            random_state=0,
            strict_mode=False,
            return_result=True,
        )

    assert result.validation["estimator_fallback_used"] is True
    assert any("'a'" in warning and "mean" in warning for warning in result.warnings)
    assert any("'b'" in warning for warning in result.warnings)


def test_typed_result_of_a_clean_run_reports_no_fallback() -> None:
    result = impute_mice(_frame(), max_iter=2, random_state=0, return_result=True)
    assert result.validation["estimator_fallback_used"] is False


def test_global_strict_mode_off_restores_the_warned_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("missingly.impute._config.strict_mode", False)
    with pytest.warns(UserWarning, match="'a'"):
        impute_mice(_frame(), estimator=_FailingRegressor(), max_iter=2, random_state=0)


def test_non_boolean_strict_mode_is_rejected() -> None:
    from missingly.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        impute_mice(_frame(), strict_mode="yes")  # type: ignore[arg-type]
