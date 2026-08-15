"""Public edge contracts for the observedness-association diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from missingly.diagnostics import missingness_association_test


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


def test_association_test_handles_sentinel_and_categorical_outcome(
) -> None:
    """Sentinel replacement and categorical factorization work with no peer features."""
    frame = pd.DataFrame({"score": [1.0, -99.0, 2.0, -99.0]})
    original = frame.copy(deep=True)

    result = missingness_association_test(
        frame,
        outcome=["late", "early", "late", "early"],
        missing_values=[-99.0],
    )

    assert result["feature"].tolist() == ["score"]
    assert result.loc[0, "n_obs"] == len(frame)
    assert result.loc[0, "lrt_statistic"] >= 0.0
    assert 0.0 <= result.loc[0, "p_value"] <= 1.0
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

    result = missingness_association_test(frame, outcome=[0, 1, 0, 1])

    assert result.empty
    assert result.columns.tolist() == [
        "feature",
        "lrt_statistic",
        "p_value",
        "n_obs",
    ]
