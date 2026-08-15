"""Public edge contracts for the observedness-association diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import (
    miss_case_summary,
    miss_var_summary,
    missingness_association_test,
)


class _ValueErrorLogisticRegression:
    """Test double that models a recoverable logistic-fit failure."""

    def __init__(self, **kwargs: object) -> None:
        """Accept the production constructor parameters.

        Parameters
        ----------
        **kwargs : object
            Ignored sklearn estimator options.

        Returns
        -------
        None
        """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ValueErrorLogisticRegression":
        """Raise a recoverable backend validation error.

        Parameters
        ----------
        X : np.ndarray
            Design matrix supplied by the diagnostic.
        y : np.ndarray
            Observedness response supplied by the diagnostic.

        Returns
        -------
        _ValueErrorLogisticRegression
            This method never returns because it raises.
        """
        raise ValueError("synthetic logistic fit failure")


class _RuntimeErrorLogisticRegression:
    """Test double that models an unexpected logistic backend failure."""

    def __init__(self, **kwargs: object) -> None:
        """Accept the production constructor parameters.

        Parameters
        ----------
        **kwargs : object
            Ignored sklearn estimator options.

        Returns
        -------
        None
        """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RuntimeErrorLogisticRegression":
        """Raise an error that the diagnostic must not hide.

        Parameters
        ----------
        X : np.ndarray
            Design matrix supplied by the diagnostic.
        y : np.ndarray
            Observedness response supplied by the diagnostic.

        Returns
        -------
        _RuntimeErrorLogisticRegression
            This method never returns because it raises.
        """
        raise RuntimeError("synthetic unexpected logistic failure")


def test_association_test_handles_sentinel_and_categorical_outcome(
) -> None:
    """Sentinel replacement and categorical factorization work with no peer features."""
    frame = pd.DataFrame({"score": [1.0, -99.0, 2.0, -99.0]})
    original = frame.copy(deep=True)

    first = missingness_association_test(
        frame,
        outcome=["late", "early", "late", "early"],
        missing_values=[-99.0],
    )
    second = missingness_association_test(
        frame,
        outcome=["late", "early", "late", "early"],
        missing_values=[-99.0],
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["feature"].tolist() == ["score"]
    assert first.loc[0, "n_obs"] == len(frame)
    assert first.loc[0, "lrt_statistic"] >= 0.0
    assert 0.0 <= first.loc[0, "p_value"] <= 1.0
    pd.testing.assert_frame_equal(frame, original)


def test_association_test_skips_constant_observedness_with_stable_schema() -> None:
    """Ineligible features leave an empty result with the documented schema."""
    frame = pd.DataFrame(
        {
            "observed": [1.0, 2.0, 3.0],
            "missing": [np.nan, np.nan, np.nan],
        }
    )

    result = missingness_association_test(frame, outcome=[0, 1, 1])

    assert result.empty
    assert result.columns.tolist() == [
        "feature",
        "lrt_statistic",
        "p_value",
        "n_obs",
    ]


def test_association_test_omits_recoverable_estimator_failures(monkeypatch) -> None:
    """A recoverable sklearn ValueError omits a feature without leaking state."""
    frame = pd.DataFrame({"score": [1.0, np.nan, 2.0, np.nan]})
    monkeypatch.setattr(
        "missingly.diagnostics.LogisticRegression",
        _ValueErrorLogisticRegression,
    )

    with pytest.warns(UserWarning, match="score"):
        result = missingness_association_test(frame, outcome=[0, 1, 0, 1])

    assert result.empty
    assert result.columns.tolist() == [
        "feature",
        "lrt_statistic",
        "p_value",
        "n_obs",
    ]


def test_association_test_propagates_unexpected_estimator_failures(monkeypatch) -> None:
    """Unexpected backend errors remain visible instead of becoming omission."""
    frame = pd.DataFrame({"score": [1.0, np.nan, 2.0, np.nan]})
    monkeypatch.setattr(
        "missingly.diagnostics.LogisticRegression",
        _RuntimeErrorLogisticRegression,
    )

    with pytest.raises(RuntimeError, match="unexpected logistic"):
        missingness_association_test(frame, outcome=[0, 1, 0, 1])


@pytest.mark.parametrize("outcome", [[0.0, np.inf, 1.0], [0.0, -np.inf, 1.0]])
def test_association_test_rejects_non_finite_numeric_outcome(
    outcome: list[float],
) -> None:
    """Numeric auxiliary outcomes must be finite before they reach sklearn."""
    frame = pd.DataFrame({"score": [1.0, np.nan, 2.0]})

    with pytest.raises(ValueError, match="finite"):
        missingness_association_test(frame, outcome=outcome)


def test_association_test_rejects_duplicate_feature_labels() -> None:
    """Duplicate labels are ambiguous for feature-wise observedness diagnostics."""
    frame = pd.DataFrame([[1.0, np.nan], [2.0, 3.0]], columns=["score", "score"])

    with pytest.raises(ValueError, match="unique"):
        missingness_association_test(frame, outcome=[0, 1])


def test_missingness_summaries_keep_documented_schema_for_empty_axes() -> None:
    """Variable and case summaries represent an empty row or column axis safely."""
    no_rows = pd.DataFrame(columns=["first", "second"])
    no_columns = pd.DataFrame(index=["first", "second"])

    variable_summary = miss_var_summary(no_rows)
    case_summary = miss_case_summary(no_columns)

    assert variable_summary.to_dict("list") == {
        "variable": ["first", "second"],
        "n_miss": [0, 0],
        "pct_miss": [0.0, 0.0],
    }
    assert case_summary.to_dict("list") == {
        "case": ["first", "second"],
        "n_miss": [0, 0],
        "pct_miss": [0.0, 0.0],
    }
