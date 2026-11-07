"""Characterisation of v1 behaviour that WP-04 must not change.

The expected column orders were recorded by running the code of
``fix/pyproject-regression`` (before the dtype layer) on mixed frames.
The model-based imputers return the numeric columns first and every other
column after them; ``impute_knn(metric="mixed")`` and the simple imputers keep
the input order.  Restoring the input order everywhere is WP-21 (F-016); here
no new order may be introduced.

Inputs the base code processed without error and that now meet an
unsupported dtype must first emit a ``FutureWarning`` and keep the old result
shape instead of raising.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    impute_gb,
    impute_knn,
    impute_mice,
    impute_polr,
    impute_polyreg,
    impute_rf,
)

_X = [1.0, np.nan, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0]
_Y = [2.0, 4.0, 6.5, 8.0, 10.0, 12.0, 14.5, 16.0]
_DAYS = pd.Series(pd.date_range("2024-01-01", periods=8))
_HOLE = pd.Series([False, True] + [False] * 6)


def _frames() -> Dict[str, pd.DataFrame]:
    return {
        "dt_full_mid": pd.DataFrame(
            {"s": list("abababab"), "when": _DAYS, "x": _X, "k": np.arange(8)}
        ),
        "dt_full_first": pd.DataFrame({"when": _DAYS, "x": _X, "y": _Y}),
        "bool_full": pd.DataFrame(
            {"s": list("abababab"), "flag": [True, False] * 4, "x": _X, "y": _Y}
        ),
        "dt_na": pd.DataFrame({"x": _X, "when": _DAYS.mask(_HOLE), "y": _Y}),
        "dttz_na": pd.DataFrame(
            {"x": _X, "when": _DAYS.dt.tz_localize("UTC").mask(_HOLE), "y": _Y}
        ),
    }


IMPUTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "knn": lambda f: impute_knn(f, n_neighbors=2),
    "knn_mixed": lambda f: impute_knn(f, n_neighbors=2, metric="mixed"),
    "mice": lambda f: impute_mice(f, max_iter=2),
    "mice_hist": lambda f: impute_mice(f, max_iter=2, return_history=True)[0],
    "rf": lambda f: impute_rf(f, n_estimators=3),
    "gb": lambda f: impute_gb(f, n_estimators=3),
    "polyreg": lambda f: impute_polyreg(f, max_iter=1),
    "polr": lambda f: impute_polr(f, max_iter=1),
}

_MODEL = ["knn", "mice", "mice_hist", "rf", "gb"]


def _expected() -> List[Tuple[str, str, List[str]]]:
    rows: List[Tuple[str, str, List[str]]] = []
    layouts = {
        "dt_full_mid": (["x", "k", "s", "when"], ["s", "when", "x", "k"]),
        "dt_full_first": (["x", "y", "when"], ["when", "x", "y"]),
        "bool_full": (["x", "y", "s", "flag"], ["s", "flag", "x", "y"]),
    }
    for frame, (numeric_first, original) in layouts.items():
        rows += [(frame, name, numeric_first) for name in _MODEL]
        rows += [(frame, name, original) for name in ("knn_mixed", "polyreg", "polr")]
    for frame in ("dt_na", "dttz_na"):
        rows += [(frame, name, ["x", "y", "when"]) for name in _MODEL]
        rows.append((frame, "knn_mixed", ["x", "when", "y"]))
    rows += [("dt_na", name, ["x", "when", "y"]) for name in ("polyreg", "polr")]
    return rows


@pytest.mark.parametrize(
    ("frame_name", "imputer", "columns"),
    [pytest.param(*row, id=f"{row[0]}-{row[1]}") for row in _expected()],
)
def test_column_order_matches_the_v1_base_code(frame_name, imputer, columns) -> None:
    frame = _frames()[frame_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = IMPUTERS[imputer](frame)
    assert list(result.columns) == columns


@pytest.mark.parametrize("frame_name", ["dt_na", "dttz_na"])
@pytest.mark.parametrize("imputer", _MODEL + ["knn_mixed"])
def test_incomplete_datetime_target_warns_instead_of_raising(
    frame_name, imputer
) -> None:
    frame = _frames()[frame_name]
    with pytest.warns(FutureWarning, match="UnsupportedDtypeError"):
        result = IMPUTERS[imputer](frame)
    assert not result["when"].isna().any()
    assert result["when"].dtype == frame["when"].dtype


@pytest.mark.parametrize("imputer", ["polyreg", "polr"])
def test_incomplete_naive_datetime_target_warns_for_the_conditional_imputers(
    imputer,
) -> None:
    frame = _frames()["dt_na"]
    with pytest.warns(FutureWarning, match="UnsupportedDtypeError"):
        IMPUTERS[imputer](frame)


def test_mice_bounds_on_a_boolean_column_warns_instead_of_raising() -> None:
    frame = pd.DataFrame(
        {"flag": [True, False, True, False], "x": [1.0, np.nan, 3.0, 4.0]}
    )
    with pytest.warns(FutureWarning, match="bounds.*boolean"):
        result = impute_mice(frame, max_iter=2, bounds={"flag": (0.0, 1.0)})
    assert not result.isna().any().any()
