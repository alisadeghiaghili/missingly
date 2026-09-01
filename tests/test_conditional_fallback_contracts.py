"""Behavioral contracts for categorical conditional-imputation fallbacks."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest
from statsmodels.miscmodels import ordinal_model

from missingly.exceptions import ImputationError
from missingly.impute import impute_logreg, impute_polr, impute_polyreg


class _ValueErrorLogisticRegression:
    """Estimator fixture that reproduces a recoverable sklearn failure."""

    def __init__(self, **kwargs) -> None:
        """Accept the production estimator constructor arguments."""

    def fit(self, X, y):
        """Raise a documented numerical-validation failure."""
        raise ValueError("synthetic singular fit")


class _RuntimeErrorLogisticRegression:
    """Estimator fixture that reproduces an unsupported backend failure."""

    def __init__(self, **kwargs) -> None:
        """Accept the production estimator constructor arguments."""

    def fit(self, X, y):
        """Raise an error that must remain visible to callers."""
        raise RuntimeError("unexpected backend failure")


class _ValueErrorOrderedModel:
    """Ordered-model fixture that reproduces a recoverable fit failure."""

    def __init__(self, endog, exog, **kwargs) -> None:
        """Accept the production ordered-model constructor arguments."""

    def fit(self, **kwargs):
        """Raise a documented numerical-validation failure."""
        raise ValueError("synthetic ordinal fit failure")


class _RuntimeErrorOrderedModel:
    """Ordered-model fixture that reproduces an unsupported backend failure."""

    def __init__(self, endog, exog, **kwargs) -> None:
        """Accept the production ordered-model constructor arguments."""

    def fit(self, **kwargs):
        """Raise an error that must remain visible to callers."""
        raise RuntimeError("unexpected ordinal backend failure")


def _nominal_frame() -> pd.DataFrame:
    """Return a nominal target with a numeric predictor and missing values."""
    target_dtype = pd.CategoricalDtype(["red", "green", "blue"], ordered=False)
    return pd.DataFrame(
        {
            "score": [0.0] * 12 + [1.0] * 12 + [2.0] * 12 + [2.0, 2.0],
            "target": pd.Series(
                ["red"] * 12 + ["green"] * 12 + ["blue"] * 12 + [pd.NA, pd.NA],
                dtype=target_dtype,
            ),
        }
    )


def _ordinal_frame() -> pd.DataFrame:
    """Return an ordered categorical target with deterministic predictors."""
    grade_dtype = pd.CategoricalDtype(["low", "medium", "high"], ordered=True)
    return pd.DataFrame(
        {
            "score": [0.0] * 12 + [1.0] * 12 + [2.0] * 12 + [2.0, 2.0],
            "grade": pd.Series(
                ["low"] * 12 + ["medium"] * 12 + ["high"] * 12 + [pd.NA, pd.NA],
                dtype=grade_dtype,
            ),
        }
    )


def _assert_preserved_observed_categories(
    original: pd.DataFrame,
    result: pd.DataFrame,
    column: str,
) -> None:
    """Assert fallback fills only missing values without changing category metadata."""
    missing_mask = original[column].isna()
    assert result[column].dtype == original[column].dtype
    pd.testing.assert_series_equal(
        result.loc[~missing_mask, column],
        original.loc[~missing_mask, column],
    )
    assert not result.loc[missing_mask, column].isna().any()


@pytest.mark.parametrize(
    ("imputer", "target", "dtype"),
    [
        (impute_polyreg, ["red", "red", "red", "green", "blue", pd.NA], "category"),
        (
            impute_polr,
            ["low", "low", "low", "medium", "high", pd.NA],
            pd.CategoricalDtype(["low", "medium", "high"], ordered=True),
        ),
    ],
    ids=["polyreg-nominal", "polr-ordinal"],
)
def test_multiclass_imputers_use_mode_without_predictors(
    imputer,
    target,
    dtype,
) -> None:
    """Use a typed modal fallback when a conditional target has no predictors.

    The standalone conditional APIs have no feature matrix to fit for a
    one-column frame. Their documented safe fallback must still fill missing
    values without mutating caller data or changing categorical metadata.
    """
    frame = pd.DataFrame({"target": pd.Series(target, dtype=dtype)})
    original = frame.copy(deep=True)

    result = imputer(frame, random_state=0)

    assert result.loc[result.index[-1], "target"] == target[0]
    _assert_preserved_observed_categories(original, result, "target")
    pd.testing.assert_frame_equal(frame, original)


def test_logreg_uses_mode_without_predictors() -> None:
    """Use the typed modal baseline for a binary target without predictors."""
    frame = pd.DataFrame(
        {"target": pd.Series(["yes", "yes", "yes", "no", pd.NA], dtype="category")}
    )
    original = frame.copy(deep=True)

    result = impute_logreg(frame, random_state=0)

    assert result.loc[result.index[-1], "target"] == "yes"
    _assert_preserved_observed_categories(original, result, "target")
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("strict_mode", [False, True], ids=["non-strict", "strict"])
def test_polr_does_not_import_a_backend_for_a_no_predictor_fallback(
    monkeypatch,
    strict_mode: bool,
) -> None:
    """Use the valid one-column mode baseline without importing a backend.

    Strict mode governs conditional-estimator errors. It must not reject this
    predictor-free baseline because no estimator is needed.
    """
    original_import = builtins.__import__

    def reject_ordinal_model_import(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        """Fail the test if the no-predictor path attempts a model import."""
        if name == "statsmodels.miscmodels.ordinal_model":
            raise AssertionError("one-column polr must not import OrderedModel")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_ordinal_model_import)
    frame = pd.DataFrame(
        {
            "grade": pd.Series(
                ["low", "low", "medium", "high", pd.NA],
                dtype=pd.CategoricalDtype(["low", "medium", "high"], ordered=True),
            )
        }
    )

    result = impute_polr(frame, random_state=0, strict_mode=strict_mode)

    assert result.loc[result.index[-1], "grade"] == "low"


def test_polyreg_recovers_from_value_error_or_raises_in_strict_mode(monkeypatch):
    """polyreg uses the mode only for supported backend failures."""
    frame = _nominal_frame()
    original = frame.copy(deep=True)
    expected_mode = frame["target"].dropna().mode().iloc[0]
    monkeypatch.setattr(
        "missingly.impute.LogisticRegression",
        _ValueErrorLogisticRegression,
    )

    with pytest.warns(UserWarning, match="Falling back to mode"):
        recovered = impute_polyreg(frame, strict_mode=False)

    assert recovered.loc[recovered.index[-2:], "target"].tolist() == [expected_mode] * 2
    _assert_preserved_observed_categories(original, recovered, "target")
    pd.testing.assert_frame_equal(frame, original)

    with pytest.raises(ImputationError) as error:
        impute_polyreg(frame, strict_mode=True)

    assert error.value.column == "target"
    assert error.value.strategy == "polyreg"
    assert isinstance(error.value.original, ValueError)


def test_polyreg_does_not_hide_unexpected_backend_errors(monkeypatch):
    """polyreg must not translate programming or unsupported backend errors."""
    monkeypatch.setattr(
        "missingly.impute.LogisticRegression",
        _RuntimeErrorLogisticRegression,
    )

    with pytest.raises(RuntimeError, match="unexpected backend failure"):
        impute_polyreg(_nominal_frame(), strict_mode=False)


def test_polr_handles_missing_statsmodels_with_multinomial_without_mutation(
    monkeypatch,
):
    """polr uses its documented sklearn fallback when statsmodels is unavailable."""
    original_import = builtins.__import__

    def import_without_ordinal_model(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        """Make only the optional ordinal-model import unavailable for this call."""
        if name == "statsmodels.miscmodels.ordinal_model":
            raise ImportError("statsmodels unavailable for this test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_ordinal_model)
    frame = _ordinal_frame()
    original = frame.copy(deep=True)

    with pytest.warns(UserWarning, match="statsmodels not found"):
        recovered = impute_polr(frame, strict_mode=False)

    _assert_preserved_observed_categories(original, recovered, "grade")
    pd.testing.assert_frame_equal(frame, original)


def test_polr_ordered_failure_falls_back_or_raises_in_strict_mode(monkeypatch):
    """polr preserves the ordered categorical contract across backend choices."""
    monkeypatch.setattr(ordinal_model, "OrderedModel", _ValueErrorOrderedModel)
    frame = _ordinal_frame()
    original = frame.copy(deep=True)

    with pytest.warns(UserWarning, match="OrderedModel failed"):
        recovered = impute_polr(frame, strict_mode=False)

    _assert_preserved_observed_categories(original, recovered, "grade")
    pd.testing.assert_frame_equal(frame, original)

    with pytest.raises(ImputationError) as error:
        impute_polr(frame, strict_mode=True)

    assert error.value.column == "grade"
    assert error.value.strategy == "polr"
    assert isinstance(error.value.original, ValueError)


def test_polr_does_not_hide_unexpected_ordered_model_errors(monkeypatch):
    """polr must not translate unsupported ordered-model backend errors."""
    monkeypatch.setattr(ordinal_model, "OrderedModel", _RuntimeErrorOrderedModel)

    with pytest.raises(RuntimeError, match="unexpected ordinal backend failure"):
        impute_polr(_ordinal_frame(), strict_mode=False)


def test_polr_uses_mode_when_both_conditional_backends_fail(monkeypatch):
    """A recoverable double backend failure falls back without altering observations."""
    monkeypatch.setattr(ordinal_model, "OrderedModel", _ValueErrorOrderedModel)
    monkeypatch.setattr(
        "missingly.impute.LogisticRegression",
        _ValueErrorLogisticRegression,
    )
    frame = _ordinal_frame()
    original = frame.copy(deep=True)
    expected_mode = frame["grade"].dropna().mode().iloc[0]

    with pytest.warns(UserWarning) as caught_warnings:
        recovered = impute_polr(frame, strict_mode=False)

    assert any(
        "multinomial fallback also failed" in str(warning.message)
        for warning in caught_warnings
    )
    assert recovered.loc[recovered.index[-2:], "grade"].tolist() == [expected_mode] * 2
    _assert_preserved_observed_categories(original, recovered, "grade")
    pd.testing.assert_frame_equal(frame, original)


def test_polr_does_not_hide_unexpected_multinomial_errors(monkeypatch):
    """polr must leave unsupported multinomial backend errors observable."""
    monkeypatch.setattr(ordinal_model, "OrderedModel", _ValueErrorOrderedModel)
    monkeypatch.setattr(
        "missingly.impute.LogisticRegression",
        _RuntimeErrorLogisticRegression,
    )

    with pytest.warns(UserWarning, match="OrderedModel failed"):
        with pytest.raises(RuntimeError, match="unexpected backend failure"):
            impute_polr(_ordinal_frame(), strict_mode=False)
