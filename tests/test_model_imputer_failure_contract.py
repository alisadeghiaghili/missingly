"""Model-based imputers must fit on predictors that contain NaN.

Random forests only accept NaN features from scikit-learn 1.4 and gradient
boosting never does, so a frame with several incomplete columns used to make
the fit fail and (with the lenient default) degrade to mean imputation.  The
predictors are now filled before every fit on every scikit-learn version.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from missingly.impute import (
    impute_gb,
    impute_logreg,
    impute_polr,
    impute_polyreg,
    impute_rf,
)

Imputer = Callable[..., pd.DataFrame]

ALL_IMPUTERS: Dict[str, Imputer] = {
    "rf": impute_rf,
    "gb": impute_gb,
    "logreg": impute_logreg,
    "polyreg": impute_polyreg,
    "polr": impute_polr,
}


def _numeric_frame(seed: int = 0, n: int = 120) -> Tuple[pd.DataFrame, pd.Series]:
    """Return correlated numeric columns whose holes sit in different rows.

    Returns
    -------
    frame, true_y
        The incomplete frame and the complete ``y`` column before the holes.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    complete = pd.DataFrame(
        {
            "x": x,
            "y": 2.0 * x + rng.normal(scale=0.1, size=n),
            "z": x + rng.normal(scale=0.5, size=n),
        }
    )
    frame = complete.copy()
    frame.loc[0::7, "x"] = np.nan
    frame.loc[3::7, "y"] = np.nan
    return frame, complete["y"]


def _frame_for(name: str) -> pd.DataFrame:
    """Return a frame with a target column suited to the named imputer."""
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "w": x + rng.normal(scale=0.3, size=n)})
    levels = {
        "logreg": ["u", "v"],
        "polyreg": ["red", "green", "blue"],
        "polr": ["low", "mid", "high"],
    }.get(name)
    if levels is None:
        frame["y"] = 2.0 * x + rng.normal(scale=0.1, size=n)
        frame.loc[1::6, "y"] = np.nan
    else:
        frame["label"] = rng.choice(levels, size=n)
        frame.loc[1::6, "label"] = np.nan
    return frame


def _target(name: str) -> str:
    return "label" if name in {"logreg", "polyreg", "polr"} else "y"



# ---------------------------------------------------------------------------
# Predictors containing NaN
# ---------------------------------------------------------------------------


def test_rf_never_hands_nan_to_the_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predictors are filled first, as scikit-learn below 1.4 rejects NaN."""
    seen_nan = []
    original_fit = RandomForestRegressor.fit

    def fit_like_old_sklearn(self, X, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_nan.append(bool(np.isnan(np.asarray(X, dtype=float)).any()))
        if seen_nan[-1]:
            raise ValueError("Input X contains NaN.")
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", fit_like_old_sklearn)
    frame, _ = _numeric_frame()

    result = impute_rf(frame, random_state=0)

    assert seen_nan and not any(seen_nan)
    assert not result.isna().any().any()
    # A column-mean fallback would repeat one value in every hole.
    assert result.loc[frame["y"].isna(), "y"].nunique() > 1


def test_gb_never_hands_nan_to_the_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_nan = []
    original_fit = GradientBoostingRegressor.fit

    def spy(self, X, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_nan.append(bool(np.isnan(np.asarray(X, dtype=float)).any()))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(GradientBoostingRegressor, "fit", spy)
    frame, _ = _numeric_frame()
    impute_gb(frame, random_state=0)
    assert seen_nan and not any(seen_nan)


@pytest.mark.parametrize("name", ["rf", "gb"])
def test_tree_imputers_recover_the_relation_between_incomplete_columns(
    name: str,
) -> None:
    """Multi-incomplete data is imputed from the other columns, not by the mean."""
    frame, true_y = _numeric_frame()
    result = ALL_IMPUTERS[name](frame, random_state=0)
    missing = frame["y"].isna().to_numpy()
    error = result.loc[missing, "y"].to_numpy() - true_y.to_numpy()[missing]
    assert float(np.sqrt(np.mean(error**2))) < 0.6 * float(true_y.std())


@pytest.mark.parametrize("name", sorted(ALL_IMPUTERS))
def test_legitimate_data_raises_and_warns_about_nothing(name: str) -> None:
    """No strategy returns fallback values silently, or at all, on valid data."""
    frame = _frame_for(name)
    frame.loc[2::11, "w"] = np.nan  # NaN in a predictor of the target
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ALL_IMPUTERS[name](frame, random_state=0)
    target = _target(name)
    assert not result[target].isna().any()
    filled = result.loc[frame[target].isna(), target]
    if name in {"rf", "gb"}:
        assert filled.nunique() > 1


# ---------------------------------------------------------------------------
# MissinglyImputer uses the same predictor handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["rf", "gb"])
def test_transformer_tree_strategies_never_hand_nan_to_the_estimator(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """fit and transform both fill incomplete predictors before the estimator."""
    from missingly.transformer import MissinglyImputer

    estimator = RandomForestRegressor if strategy == "rf" else GradientBoostingRegressor
    original_fit, original_predict = estimator.fit, estimator.predict
    seen = {"fit": [], "predict": []}

    def fit(self, X, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen["fit"].append(bool(np.isnan(np.asarray(X, dtype=float)).any()))
        return original_fit(self, X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen["predict"].append(bool(np.isnan(np.asarray(X, dtype=float)).any()))
        return original_predict(self, X, *args, **kwargs)

    monkeypatch.setattr(estimator, "fit", fit)
    monkeypatch.setattr(estimator, "predict", predict)
    frame, _ = _numeric_frame()

    imputer = MissinglyImputer(strategy=strategy, random_state=0)
    result = imputer.fit(frame).transform(frame)

    assert seen["fit"] and seen["predict"]
    assert not any(seen["fit"]) and not any(seen["predict"])
    assert not result.isna().any().any()
