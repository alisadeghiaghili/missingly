"""Unit tests for the internal dtype normalisation layer ``missingly._dtypes``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly._dtypes import (
    NUMERIC_KINDS,
    ColumnKind,
    _clip_to_integer_range,
    classify_column,
    classify_frame,
    column_mode,
    ensure_supported,
    fill_missing,
    fill_with_statistic,
    is_numeric_kind,
    normalize_frame,
    restore_dtypes,
    restore_numeric,
    to_float_array,
)
from missingly.exceptions import MissinglyError, UnsupportedDtypeError


@pytest.mark.parametrize(
    ("series", "kind"),
    [
        (pd.Series([1, 2], dtype="int64"), ColumnKind.INTEGER),
        (pd.Series([1, 2], dtype="uint8"), ColumnKind.INTEGER),
        (pd.Series([1, None], dtype="Int64"), ColumnKind.INTEGER),
        (pd.Series([1.0, 2.0], dtype="float32"), ColumnKind.FLOAT),
        (pd.Series([1.0, None], dtype="Float64"), ColumnKind.FLOAT),
        (pd.Series([True, False]), ColumnKind.BOOLEAN),
        (pd.Series([True, None], dtype="boolean"), ColumnKind.BOOLEAN),
        (pd.Series(["a", None], dtype="string"), ColumnKind.CATEGORICAL),
        (pd.Series(["a", "b"], dtype="category"), ColumnKind.CATEGORICAL),
        (pd.Series([True, False], dtype="category"), ColumnKind.CATEGORICAL),
        (pd.Series(["a", "b", None], dtype=object), ColumnKind.CATEGORICAL),
        (pd.Series([1, 2.5, None], dtype=object), ColumnKind.CATEGORICAL),
        (pd.Series([None, None], dtype=object), ColumnKind.CATEGORICAL),
        (pd.Series([1, "a", None], dtype=object), ColumnKind.MIXED),
        (pd.Series(pd.Categorical([1, "a"])), ColumnKind.MIXED),
        (pd.Series(pd.to_datetime(["2024-01-01", None])), ColumnKind.DATETIME),
        (pd.Series(pd.to_datetime(["2024-01-01"], utc=True)), ColumnKind.DATETIME),
        (pd.Series(pd.to_timedelta([1, None], unit="D")), ColumnKind.TIMEDELTA),
        (pd.Series([1 + 2j, 3j]), ColumnKind.OTHER),
        (
            pd.Series(pd.period_range("2024-01", periods=2, freq="M")),
            ColumnKind.CATEGORICAL,
        ),
    ],
)
def test_classify_column(series: pd.Series, kind: ColumnKind) -> None:
    assert classify_column(series) is kind


def test_classify_frame_preserves_order_and_numeric_kind_helper() -> None:
    frame = pd.DataFrame({"b": [1.0], "a": ["x"], "c": [True]})
    kinds = classify_frame(frame)
    assert list(kinds) == ["b", "a", "c"]
    assert [is_numeric_kind(kind) for kind in kinds.values()] == [True, False, False]
    assert NUMERIC_KINDS == {ColumnKind.INTEGER, ColumnKind.FLOAT}


def test_to_float_array_maps_na_to_nan_and_returns_a_writable_copy() -> None:
    source = pd.Series([1, None, 3], dtype="Int64")
    values = to_float_array(source)
    assert values.dtype == np.float64
    assert np.isnan(values[1])
    values[0] = 99.0  # must not write through to the source
    assert source.iloc[0] == 1


def test_column_mode_breaks_ties_by_value_or_by_first_occurrence_for_mixed() -> None:
    assert column_mode(pd.Series([3, 1, 3, 1, None])) == 1
    assert column_mode(pd.Series(["b", "a", "b", None])) == "b"
    mixed = pd.Series([2, "a", "a", 2, None], dtype=object)
    assert column_mode(mixed) == 2
    assert column_mode(mixed, ColumnKind.MIXED) == 2


def test_fill_missing_keeps_dtype_and_copies_complete_columns() -> None:
    complete = pd.Series([1, 2], dtype="Int64")
    copied = fill_missing(complete, 9)
    assert copied is not complete
    pd.testing.assert_series_equal(copied, complete)

    stamps = pd.Series(pd.to_datetime(["2024-01-01", None], utc=True))
    filled = fill_missing(stamps, stamps.iloc[0])
    assert filled.dtype == stamps.dtype
    assert not filled.isna().any()

    categories = pd.Series(
        pd.Categorical(["a", None, "b"], categories=["a", "b"], ordered=True)
    )
    filled_categories = fill_missing(categories, "b")
    assert filled_categories.dtype == categories.dtype
    assert filled_categories.tolist() == ["a", "b", "b"]


def test_fill_with_statistic_policy_for_nullable_integers() -> None:
    ints = pd.Series([1, None, 3], dtype="Int64")
    assert fill_with_statistic(ints, 2.0).dtype == "Int64"
    fractional = fill_with_statistic(ints, 2.5)
    assert fractional.dtype == "Float64"
    assert fractional.tolist() == [1.0, 2.5, 3.0]


def test_fill_with_statistic_other_kinds() -> None:
    floats = pd.Series([1.0, np.nan], dtype="float32")
    assert fill_with_statistic(floats, 2.5).dtype == np.float32
    complete = pd.Series([1.5, 2.5])
    pd.testing.assert_series_equal(fill_with_statistic(complete, 9.0), complete)
    labels = pd.Series(["a", None], dtype="string")
    assert fill_with_statistic(labels, "a").tolist() == ["a", "a"]


def test_restore_numeric_rounds_ties_to_even_and_clips_to_the_dtype_range() -> None:
    original = pd.Series([1, None, None, None], dtype="UInt8")
    imputed = pd.Series([1.0, 2.5, -4.0, 400.0])
    restored = restore_numeric(original, imputed, ColumnKind.INTEGER)
    assert restored.dtype == "UInt8"
    assert restored.tolist() == [1, 2, 0, 255]


def test_restore_numeric_preserves_observed_cells_exactly() -> None:
    big = 2**62 + 1  # not representable in float64
    original = pd.Series([big, None], dtype="Int64")
    restored = restore_numeric(
        original, pd.Series([float(big), 5.0]), ColumnKind.INTEGER
    )
    assert restored.iloc[0] == big
    assert restored.iloc[1] == 5


def test_restore_numeric_float_kinds_cast_back() -> None:
    original = pd.Series([1.0, np.nan], dtype="float32")
    restored = restore_numeric(original, pd.Series([1.0, 2.25]), ColumnKind.FLOAT)
    assert restored.dtype == np.float32
    assert restored.tolist() == [1.0, 2.25]


def test_restore_numeric_without_rounding_widens_only_when_needed() -> None:
    original = pd.Series([1, None], dtype="Int64")
    assert (
        restore_numeric(
            original, pd.Series([1.0, 2.0]), ColumnKind.INTEGER, round_integers=False
        ).dtype
        == "Int64"
    )
    assert (
        restore_numeric(
            original, pd.Series([1.0, 2.5]), ColumnKind.INTEGER, round_integers=False
        ).dtype
        == "Float64"
    )


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int64", "uint64", pd.Int32Dtype()])
def test_clip_to_integer_range_never_exceeds_the_dtype(dtype) -> None:
    clipped = _clip_to_integer_range(np.array([-1e30, 0.0, 1e30]), dtype)
    cast = pd.Series(clipped).astype(dtype)
    assert not cast.isna().any()


def test_ensure_supported_checks_only_requested_targets() -> None:
    frame = pd.DataFrame(
        {"t": pd.to_timedelta([1, None], unit="D"), "n": [1.0, np.nan]}
    )
    assert ensure_supported(frame, "knn", targets=["n"])["t"] is ColumnKind.TIMEDELTA
    with pytest.raises(UnsupportedDtypeError) as info:
        ensure_supported(frame, "knn")
    assert info.value.column == "t"
    assert info.value.strategy == "knn"
    assert isinstance(info.value, MissinglyError)
    reused = classify_frame(frame)
    assert ensure_supported(frame, "knn", targets=["n"], kinds=reused) is reused


def test_normalize_frame_splits_modelled_and_passthrough_columns() -> None:
    frame = pd.DataFrame(
        {
            "i": pd.array([1, None, 3], dtype="Int64"),
            "f": np.array([1.0, 2.0, 3.0], dtype="float32"),
            "b": pd.array([True, None, False], dtype="boolean"),
            "c": pd.Categorical(["x", None, "y"]),
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "mixed": pd.Series([1, "a", 2], dtype=object),
        }
    )
    normalized = normalize_frame(frame, "rf")
    assert list(normalized.frame.columns) == ["i", "f", "b", "c"]
    assert normalized.passthrough == ("when", "mixed")
    assert normalized.frame["i"].dtype == np.float64
    assert normalized.frame["f"].dtype == np.float64
    assert normalized.frame["b"].dtype == "boolean"
    assert normalized.frame["c"].dtype == "category"
    assert normalized.original is frame


def test_normalize_frame_raises_for_incomplete_opaque_targets_only() -> None:
    frame = pd.DataFrame(
        {"when": pd.to_datetime(["2024-01-01", None]), "x": [1.0, np.nan]}
    )
    with pytest.raises(
        UnsupportedDtypeError, match="datetime columns are not modelled"
    ):
        normalize_frame(frame, "mice")
    kept = normalize_frame(frame, "mice", targets=["x"])
    assert kept.passthrough == ("when",)


def test_restore_dtypes_copies_complete_columns_and_reattaches_passthrough() -> None:
    frame = pd.DataFrame(
        {
            "i": pd.array([1, None, 3], dtype="Int64"),
            "k": np.array([4, 5, 6], dtype="int64"),
            "s": pd.array(["a", None, "b"], dtype="string"),
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
        }
    )
    normalized = normalize_frame(frame, "knn", targets=["i", "s"])
    imputed = normalized.frame.copy()
    imputed["i"] = [1.0, 2.4, 3.0]
    imputed["k"] = [4.2, 5.2, 6.2]  # models never change observed cells
    imputed["s"] = pd.array(["a", "b", "b"], dtype="string")
    restored = restore_dtypes(imputed, normalized)
    assert list(restored.columns) == ["i", "k", "s", "when"]
    assert restored["i"].tolist() == [1, 2, 3]
    assert restored["i"].dtype == "Int64"
    pd.testing.assert_series_equal(restored["k"], frame["k"])
    assert restored["s"].tolist() == ["a", "b", "b"]
    pd.testing.assert_series_equal(restored["when"], frame["when"])


def test_restore_dtypes_without_rounding_policy() -> None:
    frame = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int64")})
    normalized = normalize_frame(frame, "mean")
    imputed = normalized.frame.copy()
    imputed["i"] = [1.0, 2.5, 3.0]
    assert (
        restore_dtypes(imputed, normalized, round_integers=False)["i"].dtype
        == "Float64"
    )
