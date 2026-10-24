"""Model-based imputers must not degrade to mean imputation without saying so.

Covers ``impute_rf``, ``impute_gb``, ``impute_logreg``, ``impute_polyreg`` and
``impute_polr``:

* predictors that contain NaN are filled before an estimator is fitted, so a
  legitimate multi-incomplete frame never triggers a fallback;
* ``strict_mode`` defaults to ``True``: a failed fit raises ``ImputationError``
  that names the column, the method and the underlying error;
* with an explicit ``strict_mode=False`` the fallback still works but emits a
  ``UserWarning`` per column that fell back.

Rejection of ``inf`` and unsupported dtypes is covered in
``test_model_imputer_input_validation.py``.
"""


from __future__ import annotations

import warnings
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression

from missingly.config import MissinglyConfig
from missingly.exceptions import ImputationError
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
# strict_mode defaults to True: forced estimator failures
# ---------------------------------------------------------------------------


def _boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    raise ValueError("forced estimator failure")


def _force_failure(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make every conditional model of *name* fail inside ``fit``."""
    if name == "rf":
        monkeypatch.setattr(RandomForestRegressor, "fit", _boom)
        monkeypatch.setattr(RandomForestClassifier, "fit", _boom)
    elif name == "gb":
        monkeypatch.setattr(GradientBoostingRegressor, "fit", _boom)
        monkeypatch.setattr(GradientBoostingClassifier, "fit", _boom)
    else:
        monkeypatch.setattr(LogisticRegression, "fit", _boom)
        if name == "polr":
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            monkeypatch.setattr(OrderedModel, "fit", _boom)


def test_strict_mode_defaults_to_true() -> None:
    assert MissinglyConfig().strict_mode is True


@pytest.mark.parametrize("name", sorted(ALL_IMPUTERS))
def test_failed_fit_raises_by_default(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _force_failure(monkeypatch, name)
    target = _target(name)

    with pytest.raises(ImputationError) as caught:
        ALL_IMPUTERS[name](_frame_for(name), random_state=0)

    error = caught.value
    assert error.column == target
    assert error.strategy == name
    assert f"column '{target}'" in str(error)
    assert f"'{name}'" in str(error)
    assert "forced estimator failure" in str(error)
    assert isinstance(error.__cause__, ValueError)


@pytest.mark.parametrize("name", ["rf", "gb"])
def test_failed_fit_warns_and_falls_back_when_strict_mode_is_off(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    frame = _frame_for(name)
    frame.loc[2::9, "x"] = np.nan  # a second incomplete column
    _force_failure(monkeypatch, name)

    with pytest.warns(UserWarning) as record:
        result = ALL_IMPUTERS[name](frame, random_state=0, strict_mode=False)

    messages = [str(item.message) for item in record if item.category is UserWarning]
    assert any("'y'" in message and name in message for message in messages)
    assert any("'x'" in message and name in message for message in messages)
    assert len(messages) == 2  # one warning per fallen-back column
    missing = frame["y"].isna()
    expected = frame["y"].mean()
    assert np.allclose(result.loc[missing, "y"], expected)
    assert not result.isna().any().any()


@pytest.mark.parametrize("name", ["logreg", "polyreg", "polr"])
def test_conditional_models_warn_per_column_when_strict_mode_is_off(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    frame = _frame_for(name)
    _force_failure(monkeypatch, name)

    with pytest.warns(UserWarning, match="'label'"):
        result = ALL_IMPUTERS[name](frame, random_state=0, strict_mode=False)

    assert not result["label"].isna().any()


def test_global_config_can_restore_the_lenient_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("missingly.impute._config.strict_mode", False)
    _force_failure(monkeypatch, "rf")
    with pytest.warns(UserWarning, match="'y'"):
        impute_rf(_frame_for("rf"), random_state=0)


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
