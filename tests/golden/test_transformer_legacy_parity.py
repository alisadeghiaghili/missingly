"""Characterisation: float64 and plain categorical inputs keep their output.

Written before the dtype handling of ``MissinglyImputer`` changed.  The frozen
pre-change implementation in :mod:`tests.golden.legacy_transformer` is the
reference, so the comparison holds on every software stack without recorded
values.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from missingly.transformer import MissinglyImputer
from tests.golden.legacy_transformer import LegacyMissinglyImputer

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

_SEED = 20261107

STRATEGY_PARAMS: Dict[str, dict] = {
    "mean": {},
    "median": {},
    "mode": {},
    "knn": {"n_neighbors": 3},
    "mice": {"max_iter": 3, "random_state": 1},
    "pmm": {"random_state": 2},
    "rf": {"random_state": 3, "estimator_kwargs": {"n_estimators": 8}},
    "gb": {"random_state": 4, "estimator_kwargs": {"n_estimators": 8}},
    "hotdeck": {"random_state": 5},
}


def _frame(rows: int, seed: int, *, with_category: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(rows, 3))
    frame = pd.DataFrame(
        {
            "a": base[:, 0] * 3.0 + 10.0,
            "b": base[:, 0] + base[:, 1],
            "c": base[:, 2] * 0.5,
        }
    )
    frame["g"] = np.where(frame["a"] > 10.0, "hi", "lo").astype(object)
    steps = [("a", 7), ("b", 5), ("c", 9), ("g", 6)]
    if with_category:
        frame["h"] = pd.Categorical(np.where(frame["b"] > 0.0, "up", "down"))
        steps.append(("h", 8))
    for column, step in steps:
        frame.loc[frame.index[step - 1 :: step], column] = np.nan
    return frame


@pytest.fixture(params=[False, True], ids=["object_only", "object_and_category"])
def frames(request: pytest.FixtureRequest) -> tuple:
    with_category = request.param
    return (
        _frame(40, _SEED, with_category=with_category),
        _frame(15, _SEED + 1, with_category=with_category),
    )


def _pair(strategy: str) -> tuple:
    params = STRATEGY_PARAMS[strategy]
    return (
        MissinglyImputer(strategy=strategy, **params),
        LegacyMissinglyImputer(strategy=strategy, **params),
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGY_PARAMS))
def test_fit_transform_matches_pre_change_output(strategy: str, frames: tuple) -> None:
    train, _ = frames
    current, legacy = _pair(strategy)
    pd.testing.assert_frame_equal(
        current.fit_transform(train), legacy.fit_transform(train), check_exact=True
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGY_PARAMS))
def test_transform_of_unseen_rows_matches_pre_change_output(
    strategy: str, frames: tuple
) -> None:
    train, test = frames
    current, legacy = _pair(strategy)
    pd.testing.assert_frame_equal(
        current.fit(train).transform(test),
        legacy.fit(train).transform(test),
        check_exact=True,
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGY_PARAMS))
def test_complete_input_matches_pre_change_output(strategy: str, frames: tuple) -> None:
    train, _ = frames
    complete = train.copy()
    complete["a"] = complete["a"].fillna(0.0)
    complete["b"] = complete["b"].fillna(0.0)
    complete["c"] = complete["c"].fillna(0.0)
    complete["g"] = complete["g"].fillna("lo")
    if "h" in complete:
        complete["h"] = complete["h"].fillna("up")
    current, legacy = _pair(strategy)
    pd.testing.assert_frame_equal(
        current.fit_transform(complete),
        legacy.fit_transform(complete),
        check_exact=True,
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGY_PARAMS))
def test_numeric_only_input_matches_pre_change_output(
    strategy: str, frames: tuple
) -> None:
    train, test = frames
    numeric_train = train[["a", "b", "c"]]
    numeric_test = test[["a", "b", "c"]]
    current, legacy = _pair(strategy)
    pd.testing.assert_frame_equal(
        current.fit(numeric_train).transform(numeric_test),
        legacy.fit(numeric_train).transform(numeric_test),
        check_exact=True,
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGY_PARAMS))
def test_single_all_missing_categorical_row_matches_pre_change_output(
    strategy: str, frames: tuple
) -> None:
    train, test = frames
    row = test.iloc[[0]].copy()
    row["g"] = np.nan
    current, legacy = _pair(strategy)
    pd.testing.assert_frame_equal(
        current.fit(train).transform(row),
        legacy.fit(train).transform(row),
        check_exact=True,
    )
