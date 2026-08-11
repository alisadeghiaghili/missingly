"""Regression tests for standalone conditional imputation functions.

The standalone API is transductive by design, but it must preserve a single
categorical feature encoding within each conditional model and honour its
documented iteration count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missingly.impute import impute_logreg, impute_pmm, impute_polyreg


def _categorical_predictor_frame(outcome: str) -> pd.DataFrame:
    """Create a categorical predictor whose ordered codes identify outcome.

    Parameters
    ----------
    outcome : {"binary", "multiclass"}
        Outcome shape to construct.

    Returns
    -------
    pd.DataFrame
        Frame with a categorical predictor and a partially missing target.

    Examples
    --------
    >>> frame = _categorical_predictor_frame("binary")
    >>> frame["target"].isna().sum()
    5
    """
    predictor = ["a"] * 12 + ["b"] * 12 + ["c"] * 12 + ["d"] * 12 + ["e"] * 12
    if outcome == "binary":
        target = ["no"] * 24 + ["yes"] * 36
    else:
        target = ["red"] * 24 + ["green"] * 24 + ["blue"] * 12

    frame = pd.DataFrame({"predictor": predictor, "target": target})
    missing_rows = pd.DataFrame({"predictor": ["e"] * 5, "target": [np.nan] * 5})
    return pd.concat([frame, missing_rows], ignore_index=True)


def test_logreg_uses_one_categorical_encoding_for_fit_and_prediction():
    """A serving-only level keeps its fit-time ordinal coordinate in logreg."""
    result = impute_logreg(_categorical_predictor_frame("binary"), random_state=0)

    assert result.loc[result.index[-5:], "target"].tolist() == ["yes"] * 5


def test_polyreg_runs_on_current_sklearn_and_reuses_feature_encoding():
    """polyreg accepts scikit-learn 1.7+ and predicts the encoded e-level."""
    result = impute_polyreg(_categorical_predictor_frame("multiclass"), random_state=0)

    assert result.loc[result.index[-5:], "target"].tolist() == ["blue"] * 5


def test_pmm_honours_max_iter(monkeypatch):
    """PMM refits the conditional model for every requested FCS sweep."""
    class CountingBayesianRidge:
        """Minimal deterministic regressor that records PMM prediction sweeps."""

        predict_calls = 0

        def fit(self, X, y):
            """Accept a fitted conditional-regression dataset.

            Parameters
            ----------
            X : np.ndarray
                Predictor matrix.
            y : np.ndarray
                Observed target values.

            Returns
            -------
            CountingBayesianRidge
                This deterministic test double.
            """
            return self

        def predict(self, X):
            """Return a fixed matching score and count one PMM sweep.

            Parameters
            ----------
            X : np.ndarray
                Predictor matrix for incomplete rows.

            Returns
            -------
            np.ndarray
                One zero score per incomplete row.
            """
            type(self).predict_calls += 1
            return np.zeros(len(X))

    monkeypatch.setattr("missingly.impute.BayesianRidge", CountingBayesianRidge)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, np.nan, 30.0, 40.0]})

    result = impute_pmm(frame, max_iter=3, random_state=0)

    assert not result["y"].isna().any()
    assert CountingBayesianRidge.predict_calls == 3
