"""``impute_mice`` has one engine: a seed fixes the result whatever else is passed.

``impute_mice`` used to dispatch to two engines. Without any optional kwarg it
delegated to scikit-learn's ``IterativeImputer``; with ``return_history``,
``return_result``, ``visit_sequence``, ``predictor_matrix``, ``where`` or
``bounds`` it ran a hand-written loop with a different random-number stream.
The same ``random_state`` therefore gave different imputations depending on
unrelated observability arguments.

Each option below is passed at its *neutral* value (the value that asks for no
change in the algorithm); the imputed values must equal those of the plain
call exactly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from missingly.impute import impute_mice


def _numeric_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 80
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "a": x,
            "b": 0.8 * x + rng.normal(scale=0.5, size=n),
            "c": rng.normal(size=n) + 0.3 * x,
        }
    )
    frame.loc[3::7, "a"] = np.nan
    frame.loc[5::9, "b"] = np.nan
    frame.loc[1::11, "c"] = np.nan
    return frame


def _mixed_frame() -> pd.DataFrame:
    frame = _numeric_frame()
    rng = np.random.default_rng(5)
    frame["g"] = rng.choice(["u", "v", "w"], size=len(frame))
    frame.loc[2::10, "g"] = np.nan
    return frame


def _neutral_options(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return every optional kwarg of ``impute_mice`` at a value that changes nothing."""
    columns = list(frame.columns)
    targets = [column for column in columns if frame[column].isna().any()]
    predictors = pd.DataFrame(
        1 - np.eye(len(columns), dtype=int), index=columns, columns=columns
    )
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    return {
        "return_history": {"return_history": True},
        "return_result": {"return_result": True},
        "visit_sequence": {"visit_sequence": targets},
        "predictor_matrix": {"predictor_matrix": predictors},
        "where": {"where": frame.isna()},
        "bounds": {"bounds": {c: (-1e9, 1e9) for c in numeric}},
        "all_options": {
            "return_history": True,
            "visit_sequence": targets,
            "predictor_matrix": predictors,
            "where": frame.isna(),
            "bounds": {c: (-1e9, 1e9) for c in numeric},
        },
    }


def _frames(output: Any, n_imputations: int) -> List[pd.DataFrame]:
    """Extract the completed data sets from any ``impute_mice`` return shape."""
    if hasattr(output, "data"):  # ImputationResult
        return list(output.data.imputations)
    if isinstance(output, tuple):  # (frame(s), history)
        output = output[0]
    return list(output) if n_imputations > 1 else [output]


_OPTIONS = [
    "return_history",
    "return_result",
    "visit_sequence",
    "predictor_matrix",
    "where",
    "bounds",
    "all_options",
]
# ``bounds`` apply to numeric targets only, so the mixed frame exercises the
# other options (and the categorical decode path they share).
_CASES = [
    pytest.param(make_frame, option, id=f"{label}-{option}")
    for make_frame, label in ((_numeric_frame, "numeric"), (_mixed_frame, "mixed"))
    for option in _OPTIONS
    if not (label == "mixed" and option in {"bounds", "all_options"})
]


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n_imputations", [1, 3], ids=["m1", "m3"])
@pytest.mark.parametrize(("make_frame", "option"), _CASES)
def test_neutral_option_does_not_change_imputed_values(
    make_frame: Callable[[], pd.DataFrame],
    option: str,
    n_imputations: int,
    seed: int,
) -> None:
    frame = make_frame()
    extra = _neutral_options(frame)[option]
    kwargs: Dict[str, Any] = {"max_iter": 4, "random_state": seed, "n_imputations": n_imputations}

    plain = _frames(impute_mice(frame, **kwargs), n_imputations)
    with_option = _frames(impute_mice(frame, **kwargs, **extra), n_imputations)

    assert len(plain) == len(with_option) == n_imputations
    for expected, actual in zip(plain, with_option):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), RandomForestRegressor(n_estimators=5)],
    ids=["linear", "forest"],
)
@pytest.mark.parametrize("option", ["return_history", "visit_sequence", "where"])
def test_custom_estimator_result_does_not_depend_on_options(
    estimator: Any, option: str
) -> None:
    frame = _numeric_frame()
    extra = _neutral_options(frame)[option]
    kwargs: Dict[str, Any] = {"max_iter": 3, "random_state": 4, "estimator": estimator}

    plain = _frames(impute_mice(frame, **kwargs), 1)[0]
    with_option = _frames(impute_mice(frame, **kwargs, **extra), 1)[0]

    pd.testing.assert_frame_equal(with_option, plain, check_exact=True)


def test_same_seed_reproduces_and_other_seed_differs() -> None:
    frame = _numeric_frame()
    first = impute_mice(frame, random_state=3)
    again = impute_mice(frame, random_state=3)
    other = impute_mice(frame, random_state=4)
    pd.testing.assert_frame_equal(first, again, check_exact=True)
    assert not first.equals(other)


@pytest.mark.parametrize("n_imputations", [1, 3], ids=["m1", "m3"])
def test_history_ends_at_the_mean_of_the_returned_imputations(n_imputations: int) -> None:
    frame = _numeric_frame()
    max_iter = 5

    completed, histories = impute_mice(
        frame,
        max_iter=max_iter,
        random_state=2,
        n_imputations=n_imputations,
        return_history=True,
    )
    if n_imputations == 1:
        completed, histories = [completed], [histories]

    assert len(completed) == len(histories) == n_imputations
    for frame_i, history in zip(completed, histories):
        assert set(history) == {c for c in frame.columns if frame[c].isna().any()}
        for column, trace in history.items():
            assert len(trace) == max_iter
            missing = frame[column].isna()
            assert trace[-1] == float(frame_i.loc[missing, column].mean())


def test_returned_history_and_result_contract_agree() -> None:
    """``return_result`` records the same traces that ``return_history`` returns."""
    frame = _numeric_frame()
    kwargs: Dict[str, Any] = {"max_iter": 4, "random_state": 9, "n_imputations": 2}

    _, histories = impute_mice(frame, return_history=True, **kwargs)
    typed = impute_mice(frame, return_result=True, **kwargs)

    for legacy, recorded in zip(histories, typed.histories):
        assert {k: tuple(v) for k, v in legacy.items()} == dict(recorded)


def test_where_mask_keeps_unobserved_cells_out_of_the_training_rows() -> None:
    """A cell retained as missing is not an observation of its column.

    Cells that ``where`` leaves unimputed used to be seeded with the column
    mean and then used as training targets for the column's own model.
    """
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": 3.0 * x + rng.normal(scale=0.05, size=n)})
    frame.loc[0:19, "y"] = np.nan
    where = frame.isna()
    where.loc[0:14, "y"] = False  # 15 of the 20 holes are retained as missing
    optional: Optional[pd.DataFrame] = where

    result = impute_mice(frame, max_iter=5, random_state=0, where=optional)

    filled = result.loc[15:19, "y"].to_numpy()
    truth = 3.0 * x[15:20]
    # With the mean-seeded retained cells as training rows the slope is biased
    # towards zero; with the 40 genuinely observed rows it is recovered.
    assert np.max(np.abs(filled - truth)) < 0.5
    assert result.loc[0:14, "y"].isna().all()
