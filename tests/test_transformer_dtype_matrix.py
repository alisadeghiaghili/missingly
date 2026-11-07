"""Dtype contract of :class:`missingly.transformer.MissinglyImputer`.

Every strategy is run over one extra "subject" column of each pandas dtype
family, with and without missing values, through ``fit``, ``transform`` and
``fit_transform``.  The contracts mirror the ones of the functional imputers
(``tests/test_dtype_normalisation_matrix.py``):

1. no raw crash: a call succeeds, or raises the single documented
   :class:`~missingly.exceptions.UnsupportedDtypeError` for an opaque dtype the
   strategy does not model, or warns with ``FutureWarning`` for the datetime
   case the previous release imputed as a nominal category;
2. columns that need no imputation keep their exact dtype and values;
3. imputed columns get their original dtype back (the documented exception:
   mean / median of a nullable integer column is ``Float64`` when the fill is
   not integral);
4. nothing else emits a ``FutureWarning``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import InsufficientDataError, UnsupportedDtypeError
from missingly.transformer import MissinglyImputer
from tests.test_dtype_normalisation_matrix import (
    OPAQUE_DTYPES,
    SUBJECT_DTYPES,
    make_frame,
)

pytestmark = [
    pytest.mark.filterwarnings("error::FutureWarning"),
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning"),
]

STRATEGY_PARAMS = {
    "mean": {},
    "median": {},
    "mode": {},
    "knn": {"n_neighbors": 3},
    "mice": {"max_iter": 3},
    "pmm": {},
    "rf": {"estimator_kwargs": {"n_estimators": 5}},
    "gb": {"estimator_kwargs": {"n_estimators": 5}},
    "hotdeck": {},
}
STRATEGIES = sorted(STRATEGY_PARAMS)
MODEL_STRATEGIES = frozenset({"knn", "mice", "rf", "gb"})
DATETIME_DTYPES = frozenset({"datetime64", "datetime_tz"})
DTYPES = list(SUBJECT_DTYPES)


def make_imputer(strategy: str) -> MissinglyImputer:
    return MissinglyImputer(
        strategy=strategy, random_state=0, **STRATEGY_PARAMS[strategy]
    )


def expected_outcome(strategy: str, dtype: str, with_missing: bool) -> str:
    """Return ``ok``, ``warn`` (FutureWarning, keeps working) or ``error``."""
    if (
        not with_missing
        or strategy not in MODEL_STRATEGIES
        or dtype not in OPAQUE_DTYPES
    ):
        return "ok"
    return "warn" if dtype in DATETIME_DTYPES else "error"


def run_with_outcome(outcome: str, call):
    """Run *call* under the expected outcome and return its result (or ``None``)."""
    if outcome == "error":
        with pytest.raises(UnsupportedDtypeError) as caught:
            call()
        assert caught.value.column == "s"
        return None
    if outcome == "warn":
        with pytest.warns(FutureWarning, match="deprecated"):
            return call()
    return call()


def assert_contract(
    result: pd.DataFrame, frame: pd.DataFrame, strategy: str, dtype: str
) -> None:
    assert list(result.columns) == list(frame.columns)
    assert len(result) == len(frame)
    assert not result.isna().any().any()
    for column in frame.columns:
        original, got = frame[column], result[column]
        missing = original.isna().to_numpy()
        if not missing.any():
            pd.testing.assert_series_equal(got, original)
            continue
        allowed = {original.dtype}
        if strategy in ("mean", "median") and dtype == "Int64" and column == "s":
            allowed.add(pd.Float64Dtype())
        assert got.dtype in allowed, (column, got.dtype, original.dtype)
        kept = ~missing
        assert (
            got[kept].astype(object).tolist() == original[kept].astype(object).tolist()
        )


@pytest.mark.parametrize("with_missing", [False, True], ids=["complete", "missing"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strategy", STRATEGIES)
class TestMatrix:
    def test_fit(self, strategy: str, dtype: str, with_missing: bool) -> None:
        frame = make_frame(dtype, with_missing)
        outcome = expected_outcome(strategy, dtype, with_missing)
        fitted = run_with_outcome(outcome, lambda: make_imputer(strategy).fit(frame))
        if outcome != "error":
            assert isinstance(fitted, MissinglyImputer)

    def test_transform(self, strategy: str, dtype: str, with_missing: bool) -> None:
        frame = make_frame(dtype, with_missing)
        outcome = expected_outcome(strategy, dtype, with_missing)

        def call() -> pd.DataFrame:
            return make_imputer(strategy).fit(frame).transform(frame)

        result = run_with_outcome(outcome, call)
        if outcome != "error":
            assert_contract(result, frame, strategy, dtype)

    def test_fit_transform(self, strategy: str, dtype: str, with_missing: bool) -> None:
        frame = make_frame(dtype, with_missing)
        outcome = expected_outcome(strategy, dtype, with_missing)
        result = run_with_outcome(
            outcome, lambda: make_imputer(strategy).fit_transform(frame)
        )
        if outcome == "error":
            return
        assert_contract(result, frame, strategy, dtype)
        separate = run_with_outcome(
            outcome, lambda: make_imputer(strategy).fit(frame).transform(frame)
        )
        pd.testing.assert_frame_equal(result, separate)


@pytest.mark.parametrize("dtype", [d for d in DTYPES if SUBJECT_DTYPES[d]])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_serving_rows_with_missing_after_complete_training(
    strategy: str, dtype: str
) -> None:
    """Fit on a complete subject column, then serve a frame that has holes."""
    train = make_frame(dtype, with_missing=False)
    serve = make_frame(dtype, with_missing=True)
    imputer = make_imputer(strategy).fit(train)
    opaque_model = strategy in MODEL_STRATEGIES and dtype in OPAQUE_DTYPES
    if opaque_model and dtype not in DATETIME_DTYPES:
        with pytest.raises(UnsupportedDtypeError) as caught:
            imputer.transform(serve)
        assert caught.value.column == "s"
        return
    if opaque_model:
        # The previous release imputed it as a nominal category: warn, keep working.
        with pytest.warns(FutureWarning, match="deprecated"):
            result = imputer.transform(serve)
    else:
        result = imputer.transform(serve)
    assert_contract(result, serve, strategy, dtype)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_complete_opaque_columns_are_not_predictors(strategy: str) -> None:
    """A complete datetime column passes through and never alters other columns."""
    frame = make_frame("datetime64", with_missing=False)
    without = frame.drop(columns="s")
    with_stamps = make_imputer(strategy).fit_transform(frame)
    plain = make_imputer(strategy).fit_transform(without)
    pd.testing.assert_series_equal(with_stamps["s"], frame["s"])
    if strategy in ("pmm", "hotdeck", "mean", "median", "mode"):
        pd.testing.assert_frame_equal(with_stamps.drop(columns="s"), plain)


@pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
@pytest.mark.parametrize("dtype", ["Int64", "string", "boolean", "float32"])
def test_all_missing_column_raises_insufficient_data(strategy: str, dtype: str) -> None:
    frame = make_frame(dtype, with_missing=False)
    frame["s"] = pd.Series([None] * len(frame), dtype=frame["s"].dtype)
    with pytest.raises(InsufficientDataError):
        make_imputer(strategy).fit(frame)


# ---------------------------------------------------------------- equivalence

_EQ_INTS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7, 8, 2, 4, 6, 9, 1, 5, 3, 7]


def equivalence_frames(nullable: str) -> pd.DataFrame:
    """A frame with an integer-valued incomplete column ``u`` and a float one ``v``."""
    rng = np.random.default_rng(11)
    u = [float(v) for v in _EQ_INTS]
    v = list(rng.normal(size=len(u)) + np.asarray(u))
    for position in (2, 9, 15):
        u[position] = np.nan
    for position in (4, 11):
        v[position] = np.nan
    frame = pd.DataFrame({"u": u, "v": v, "w": rng.normal(size=len(u))})
    if nullable != "float64":
        frame["u"] = frame["u"].astype(nullable)
    return frame


ROUNDING_STRATEGIES = ["knn", "mice", "pmm", "rf", "gb"]


@pytest.mark.parametrize("nullable", ["Int64", "Float64", "float32"])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_nullable_input_matches_float64_input(strategy: str, nullable: str) -> None:
    plain = equivalence_frames("float64")
    other = equivalence_frames(nullable)
    expected = make_imputer(strategy).fit_transform(plain)
    got = make_imputer(strategy).fit_transform(other)
    assert got["u"].dtype == other["u"].dtype or (
        strategy in ("mean", "median") and nullable == "Int64"
    )
    u_expected = expected["u"].to_numpy(dtype="float64")
    if nullable == "Int64" and strategy in ROUNDING_STRATEGIES:
        u_expected = np.rint(u_expected)
    if nullable == "float32":
        u_expected = u_expected.astype("float32").astype("float64")
    np.testing.assert_allclose(
        got["u"].to_numpy(dtype="float64"), u_expected, rtol=1e-6
    )
    pd.testing.assert_frame_equal(
        got[["v", "w"]], expected[["v", "w"]], check_exact=False, rtol=1e-6
    )


@pytest.mark.parametrize("strategy", ROUNDING_STRATEGIES)
def test_integer_targets_are_rounded_not_truncated(strategy: str) -> None:
    frame = equivalence_frames("Int64")
    result = make_imputer(strategy).fit_transform(frame)
    assert result["u"].dtype == pd.Int64Dtype()
    assert result["u"].notna().all()
    assert 1 <= int(result["u"].min()) and int(result["u"].max()) <= 9


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_mean_and_median_do_not_round_nullable_integers(strategy: str) -> None:
    frame = pd.DataFrame({"u": pd.array([1, 2, None, 2], dtype="Int64")})
    result = make_imputer(strategy).fit_transform(frame)
    fill = 5 / 3 if strategy == "mean" else 2.0
    assert result["u"].tolist()[2] == pytest.approx(fill)
    expected = pd.Float64Dtype() if strategy == "mean" else pd.Int64Dtype()
    assert result["u"].dtype == expected


# ------------------------------------------------------------------- F-043


def _tree_frame(target: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = len(target)
    return pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "t": target}
    )


@pytest.mark.parametrize("strategy", ["rf", "gb"])
def test_tree_strategies_impute_an_int64_target(strategy: str) -> None:
    target = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype="Int64")
    target.iloc[[3, 8]] = pd.NA
    frame = _tree_frame(target)
    result = make_imputer(strategy).fit_transform(frame)
    assert result["t"].dtype == pd.Int64Dtype()
    assert result["t"].notna().all()
    assert result["t"].iloc[[0, 1, 2]].tolist() == [1, 2, 3]


@pytest.mark.parametrize("strategy", ["rf", "gb"])
def test_tree_strategies_return_a_boolean_target_as_boolean(strategy: str) -> None:
    flags = [True, False] * 6
    target = pd.Series(flags, dtype="boolean")
    target.iloc[[3, 8]] = pd.NA
    frame = _tree_frame(target)
    result = make_imputer(strategy).fit_transform(frame)
    assert result["t"].dtype == pd.BooleanDtype()
    assert set(result["t"].tolist()) <= {True, False}
    assert result["t"].iloc[[0, 1, 2]].tolist() == [True, False, True]


@pytest.mark.parametrize("strategy", ["rf", "gb"])
def test_tree_strategies_keep_float32_and_float64_targets(strategy: str) -> None:
    for dtype in ("float32", "Float64"):
        target = pd.Series(np.linspace(0.5, 6.0, 12), dtype=dtype)
        target.iloc[[3, 8]] = np.nan
        result = make_imputer(strategy).fit_transform(_tree_frame(target))
        assert result["t"].dtype == target.dtype


# ---------------------------------------------------- serving-time conveniences


def test_all_missing_numeric_serving_row_stays_numeric() -> None:
    train = make_frame("Int64", with_missing=True)
    row = train.iloc[[0]].copy()
    row["x"] = np.nan
    row["s"] = pd.array([None], dtype="Int64")
    result = make_imputer("knn").fit(train).transform(row)
    assert result["s"].dtype == pd.Int64Dtype()
    assert result["s"].notna().all()


def test_object_column_of_none_is_numeric_when_the_fit_was_numeric() -> None:
    train = pd.DataFrame({"age": [25.0, np.nan, 35.0]})
    result = make_imputer("mean").fit(train).transform(pd.DataFrame({"age": [None]}))
    assert result["age"].dtype == np.dtype("float64")
    assert float(result.loc[0, "age"]) == 30.0


def test_dtype_notes_are_documented() -> None:
    doc: Optional[str] = MissinglyImputer.__doc__
    assert doc is not None and "Dtype handling" in doc
