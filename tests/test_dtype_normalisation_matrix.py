"""Dtype-normalisation contract for every public imputer in ``missingly.impute``.

The matrix runs each imputer over one extra "subject" column of every pandas
dtype family, with and without missing values, and checks four contracts:

1. no raw crash: either the call succeeds or it raises the single documented
   ``UnsupportedDtypeError`` for a dtype the imputer does not model;
2. columns that need no imputation keep their exact dtype and values;
3. imputed columns get their original dtype back (the only documented
   exception: mean / median of a nullable integer column is returned as
   ``Float64`` because a mean is not an integer);
4. nothing in the matrix emits a ``FutureWarning`` (pandas treats those as
   future errors).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import InsufficientDataError, MissinglyError
from missingly.impute import (
    FittedImputer,
    impute_gb,
    impute_knn,
    impute_logreg,
    impute_mean,
    impute_median,
    impute_mice,
    impute_mode,
    impute_pmm,
    impute_polr,
    impute_polyreg,
    impute_rf,
)

pytestmark = pytest.mark.filterwarnings("error::FutureWarning")

N_ROWS = 10
MISSING_POSITIONS = (3, 7)

_STRINGS = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
_FLOATS = [2.5, 1.0, 4.0, 3.5, 6.0, 5.5, 8.0, 7.0, 9.5, 9.0]
_INTS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
_BOOLS = [True, False, True, True, False, False, True, False, True, False]
_MIXED = [1, "a", 2, "b", 3, "a", 1, "b", 2, "a"]
_DAYS = [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]

# dtype family -> can the dtype hold a missing value?
SUBJECT_DTYPES: Dict[str, bool] = {
    "int64": False,
    "float32": True,
    "Int64": True,
    "Float64": True,
    "boolean": True,
    "bool": False,
    "string": True,
    "category": True,
    "object_str": True,
    "datetime64": True,
    "datetime_tz": True,
    "timedelta64": True,
    "object_mixed": True,
}
OPAQUE_DTYPES = {"datetime64", "datetime_tz", "timedelta64", "object_mixed"}


def _with_holes(values: list, with_missing: bool) -> list:
    out = list(values)
    if with_missing:
        for position in MISSING_POSITIONS:
            out[position] = None
    return out


def make_subject(name: str, with_missing: bool) -> pd.Series:
    """Build the subject column for one dtype family."""
    if name == "int64":
        return pd.Series(_INTS, dtype="int64")
    if name == "bool":
        return pd.Series(_BOOLS, dtype="bool")

    def holes(values: list) -> list:
        return _with_holes(values, with_missing)

    if name == "float32":
        return pd.Series(holes(_FLOATS), dtype="float32")
    if name == "Int64":
        return pd.Series(holes(_INTS), dtype="Int64")
    if name == "Float64":
        return pd.Series(holes(_FLOATS), dtype="Float64")
    if name == "boolean":
        return pd.Series(holes(_BOOLS), dtype="boolean")
    if name == "string":
        return pd.Series(holes(_STRINGS), dtype="string")
    if name == "category":
        return pd.Series(pd.Categorical(holes(_STRINGS)))
    if name == "object_str":
        return pd.Series(holes(_STRINGS), dtype=object)
    if name == "datetime64":
        stamps = pd.to_datetime(
            [pd.Timestamp("2024-01-01") + pd.Timedelta(days=d) for d in _DAYS]
        )
        return pd.Series(pd.DatetimeIndex(stamps)).mask(
            pd.Series([i in MISSING_POSITIONS and with_missing for i in range(N_ROWS)])
        )
    if name == "datetime_tz":
        stamps = pd.to_datetime(
            [pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d) for d in _DAYS]
        )
        return pd.Series(pd.DatetimeIndex(stamps)).mask(
            pd.Series([i in MISSING_POSITIONS and with_missing for i in range(N_ROWS)])
        )
    if name == "timedelta64":
        return pd.Series(pd.to_timedelta(holes(_DAYS), unit="D"))
    if name == "object_mixed":
        return pd.Series(holes(_MIXED), dtype=object)
    raise AssertionError(name)


def make_frame(name: str, with_missing: bool) -> pd.DataFrame:
    """Two float64 columns (``x`` incomplete) plus the subject column ``s``."""
    x = [1.0, 2.0, np.nan, 4.0, 5.5, 6.0, np.nan, 8.5, 9.0, 10.0]
    y = [2.0, 4.5, 6.0, 7.5, 10.0, 12.5, 13.0, 17.0, 18.0, 20.5]
    frame = pd.DataFrame({"x": x, "y": y})
    frame["s"] = make_subject(name, with_missing)
    return frame


def _history_first(result):
    return result[0]


def _mice(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return impute_mice(frame, max_iter=2, random_state=0, **kwargs)


def _targets(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if frame[c].isna().any()]


def _predictor_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    k = len(frame.columns)
    return pd.DataFrame(1 - np.eye(k, dtype=int), index=frame.columns, columns=frame.columns)


def _fitted(strategy: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def run(frame: pd.DataFrame) -> pd.DataFrame:
        return FittedImputer(strategy).fit_transform(frame)

    return run


def _fitted_split(frame: pd.DataFrame) -> pd.DataFrame:
    imputer = FittedImputer("mean").fit(frame)
    return imputer.transform(frame)


IMPUTERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "mean": impute_mean,
    "median": impute_median,
    "mode": impute_mode,
    "knn": lambda f: impute_knn(f, n_neighbors=2),
    "knn_mixed": lambda f: impute_knn(f, n_neighbors=2, metric="mixed"),
    "mice": lambda f: _mice(f),
    "mice_history": lambda f: _history_first(_mice(f, return_history=True)),
    "mice_where": lambda f: _mice(f, where=f.isna()),
    "mice_bounds": lambda f: _mice(f, bounds={"x": (-100.0, 100.0)}),
    "mice_visit": lambda f: _mice(f, visit_sequence=_targets(f)),
    "mice_predictors": lambda f: _mice(f, predictor_matrix=_predictor_matrix(f)),
    "mice_result": lambda f: _mice(f, return_result=True).data.imputations[0],
    "pmm": lambda f: impute_pmm(f, max_iter=2, random_state=0),
    "rf": lambda f: impute_rf(f, random_state=0, n_estimators=5),
    "gb": lambda f: impute_gb(f, random_state=0, n_estimators=5),
    "logreg": lambda f: impute_logreg(f, max_iter=2),
    "polyreg": lambda f: impute_polyreg(f, max_iter=2),
    "polr": lambda f: impute_polr(f, max_iter=2),
    "fitted_mean": _fitted("mean"),
    "fitted_median": _fitted("median"),
    "fitted_mode": _fitted("mode"),
    "fitted_split": _fitted_split,
}

MODEL_IMPUTERS = {
    "knn", "knn_mixed", "mice", "mice_history", "mice_where", "mice_bounds",
    "mice_visit", "mice_predictors", "mice_result", "rf", "gb",
}
CONDITIONAL_IMPUTERS = {"logreg", "polyreg", "polr"}
REAL_VALUED_IMPUTERS = {"mean", "median", "fitted_mean", "fitted_median", "fitted_split"}
KEEP_COLUMN_ORDER = set(IMPUTERS) - MODEL_IMPUTERS | {"mice_result"}

CASES = [
    pytest.param(imputer, dtype, with_missing, id=f"{imputer}-{dtype}-{'na' if with_missing else 'full'}")
    for imputer in IMPUTERS
    for dtype, can_hold_na in SUBJECT_DTYPES.items()
    for with_missing in ((False, True) if can_hold_na else (False,))
]


def _expects_unsupported(imputer: str, dtype: str, with_missing: bool) -> bool:
    return imputer in MODEL_IMPUTERS and dtype in OPAQUE_DTYPES and with_missing


def _expected_dtype(imputer: str, original) -> object:
    if imputer in REAL_VALUED_IMPUTERS and str(original) == "Int64":
        return pd.Float64Dtype()
    return original


def _run(imputer: str, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    return IMPUTERS[imputer](frame)


@pytest.mark.parametrize(("imputer", "dtype", "with_missing"), CASES)
def test_imputer_dtype_matrix(imputer: str, dtype: str, with_missing: bool) -> None:
    frame = make_frame(dtype, with_missing)
    original = frame.copy(deep=True)

    if _expects_unsupported(imputer, dtype, with_missing):
        with pytest.raises(MissinglyError) as excinfo:
            _run(imputer, frame)
        assert type(excinfo.value).__name__ == "UnsupportedDtypeError"
        assert "s" in str(excinfo.value)
        pd.testing.assert_frame_equal(frame, original)
        return

    result = _run(imputer, frame)

    pd.testing.assert_frame_equal(frame, original)  # the input is never mutated
    assert set(result.columns) == set(frame.columns)
    if imputer in KEEP_COLUMN_ORDER:
        assert list(result.columns) == list(frame.columns)

    for column in frame.columns:
        if not frame[column].isna().any():
            pd.testing.assert_series_equal(
                result[column], frame[column], check_exact=True, check_names=True
            )

    if imputer in CONDITIONAL_IMPUTERS:
        return  # these only target binary / polytomous columns (WP-23 scope)

    assert not result.isna().any().any()
    assert result["x"].dtype == frame["x"].dtype
    assert result["s"].dtype == _expected_dtype(imputer, frame["s"].dtype)


@pytest.mark.parametrize("imputer", sorted(MODEL_IMPUTERS | {"pmm", "mean", "median", "mode"}))
@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_frame_matches_float64_frame(imputer: str, dtype: str) -> None:
    nullable = make_frame(dtype, True)[["x", "y", "s"]]
    reference = nullable.astype("float64")

    got = _run(imputer, nullable)
    want = _run(imputer, reference)

    got_values = got.astype("float64")[list(want.columns)]
    if dtype == "Int64" and imputer in MODEL_IMPUTERS:
        want = want.copy()
        want["s"] = np.rint(want["s"])
    pd.testing.assert_frame_equal(got_values, want, check_exact=True)


@pytest.mark.parametrize("imputer", sorted(set(IMPUTERS) - CONDITIONAL_IMPUTERS - {"mode"}))
@pytest.mark.parametrize("dtype", ["float32", "Int64", "string", "category"])
def test_all_missing_column_raises_insufficient_data(imputer: str, dtype: str) -> None:
    frame = make_frame(dtype, False)
    frame["s"] = pd.Series([None] * N_ROWS, dtype=frame["s"].dtype)
    with pytest.raises(InsufficientDataError):
        _run(imputer, frame)


def test_mode_all_missing_column_raises_insufficient_data() -> None:
    frame = make_frame("Float64", False)
    frame["s"] = pd.Series([None] * N_ROWS, dtype="Float64")
    with pytest.raises(InsufficientDataError):
        impute_mode(frame)


@pytest.mark.parametrize("imputer", sorted(set(IMPUTERS) - CONDITIONAL_IMPUTERS))
def test_single_observed_value_is_defined(imputer: str) -> None:
    frame = make_frame("float32", False)
    frame["s"] = pd.Series([4.0] + [None] * (N_ROWS - 1), dtype="float32")
    try:
        result = _run(imputer, frame)
    except MissinglyError:
        return  # a clear, typed refusal is an accepted outcome
    assert not result.isna().any().any()
    np.testing.assert_allclose(result["s"].to_numpy(dtype="float64"), 4.0, atol=1e-3)


def test_mixed_object_column_is_mode_filled_by_simple_imputers() -> None:
    frame = make_frame("object_mixed", True)
    for strategy in ("mean", "median", "mode"):
        result = FittedImputer(strategy).fit_transform(frame)
        assert result["s"].isna().sum() == 0
        assert result["s"].tolist()[0] == 1
