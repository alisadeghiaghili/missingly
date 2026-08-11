"""Regression tests for standalone conditional imputation functions.

The standalone API is transductive by design, but it must preserve a single
categorical feature encoding within each conditional model and honour its
documented iteration count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missingly.impute import impute_logreg, impute_pmm, impute_polr, impute_polyreg


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


def test_polr_uses_declared_ordinal_category_order():
    """polr fits OrderedModel on ordered categories rather than object labels."""
    frame = pd.DataFrame(
        {
            "score": [0.0] * 20 + [1.0] * 20 + [2.0] * 25,
            "grade": pd.Categorical(
                ["low"] * 20 + ["medium"] * 20 + ["high"] * 20 + [pd.NA] * 5,
                categories=["low", "medium", "high"],
                ordered=True,
            ),
        }
    )

    result = impute_polr(frame, max_iter=1, random_state=0)

    assert result.loc[result.index[-5:], "grade"].tolist() == ["high"] * 5


def test_pmm_honours_max_iter(monkeypatch):
    """PMM refits the conditional model for every requested FCS sweep."""
    class CountingBayesianRidge:
        """Minimal deterministic regressor that records PMM prediction sweeps."""

        predict_calls = 0
        coef_ = np.array([0.0])
        sigma_ = np.zeros((1, 1))
        intercept_ = 0.0

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


def test_pmm_matches_predicted_donor_scores_not_observed_values(monkeypatch):
    """PMM chooses donors by predictive distance, not distance to raw outcomes."""
    class ScoreBayesianRidge:
        """Deterministic posterior model whose score is the sole feature value."""

        coef_ = np.array([1.0])
        sigma_ = np.zeros((1, 1))
        intercept_ = 0.0

        def fit(self, X, y):
            """Accept a conditional-regression training set.

            Parameters
            ----------
            X : np.ndarray
                Observed predictor values.
            y : np.ndarray
                Observed target values.

            Returns
            -------
            ScoreBayesianRidge
                This deterministic test double.
            """
            return self

        def predict(self, X):
            """Return the conditional predictive mean for each row.

            Parameters
            ----------
            X : np.ndarray
                Predictor values.

            Returns
            -------
            np.ndarray
                The first predictor column as PMM matching scores.
            """
            return X[:, 0]

    monkeypatch.setattr("missingly.impute.BayesianRidge", ScoreBayesianRidge)
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 2.0], "y": [100.0, 200.0, 300.0, np.nan]}
    )

    result = impute_pmm(frame, max_iter=1, n_nearest_donors=1, random_state=0)

    assert result.loc[3, "y"] == 300.0
