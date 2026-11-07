"""Edge contracts of the dtype handling in ``missingly.impute`` (WP-04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    _assign_labels,
    _fill_categorical_gaps,
    _missing_to_nan,
    _require_observed,
    impute_gb,
    impute_knn,
    impute_mean,
    impute_mice,
    impute_pmm,
    impute_rf,
)

pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _opaque_only_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "gap": pd.to_timedelta([1, 2, 3], unit="D"),
            "mixed": pd.Series([1, "a", 2], dtype=object),
        }
    )


@pytest.mark.parametrize("metric", ["euclidean", "mixed"])
def test_knn_on_a_frame_without_modelled_columns_returns_an_exact_copy(
    metric: str,
) -> None:
    frame = _opaque_only_frame()
    result = impute_knn(frame, n_neighbors=2, metric=metric)
    pd.testing.assert_frame_equal(result, frame)
    assert result is not frame


@pytest.mark.parametrize("imputer", [impute_rf, impute_gb, impute_pmm])
def test_other_imputers_accept_a_frame_without_modelled_columns(imputer) -> None:
    frame = _opaque_only_frame()
    pd.testing.assert_frame_equal(imputer(frame)[list(frame.columns)], frame)


def test_mice_on_a_frame_without_modelled_columns_returns_an_exact_copy() -> None:
    frame = _opaque_only_frame()
    pd.testing.assert_frame_equal(impute_mice(frame)[list(frame.columns)], frame)
    result, history = impute_mice(frame, return_history=True)
    pd.testing.assert_frame_equal(result[list(frame.columns)], frame)
    assert history == {}


def test_mice_keeps_a_retained_opaque_cell_missing_instead_of_raising() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0, 5.0],
            "when": pd.to_datetime(
                ["2024-01-01", None, "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )
    where = pd.DataFrame({"x": [False, True, False, False, False], "when": [False] * 5})
    result = impute_mice(frame, where=where, max_iter=2)
    assert result["when"].isna().tolist() == [False, True, False, False, False]
    assert not result["x"].isna().any()
    with pytest.warns(FutureWarning, match="UnsupportedDtypeError"):
        impute_mice(frame, max_iter=2)


def test_mice_where_restores_nullable_integer_and_retains_missing_cells() -> None:
    frame = pd.DataFrame(
        {
            "n": pd.array([1, None, 3, None, 5, 6], dtype="Int64"),
            "x": [1.0, 2.0, 3.0, 4.5, 5.0, 6.5],
        }
    )
    where = pd.DataFrame(
        {"n": [False, True, False, False, False, False], "x": [False] * 6}
    )
    result = impute_mice(frame, where=where, max_iter=2)
    assert result["n"].dtype == "Int64"
    assert result["n"].isna().tolist() == [False, False, False, True, False, False]


def test_mice_bounds_reject_non_numeric_columns() -> None:
    frame = pd.DataFrame({"label": ["a", "b", "a"], "x": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="must be numeric"):
        impute_mice(frame, bounds={"label": (0.0, 1.0)})


def test_mice_return_result_accepts_interleaved_column_kinds() -> None:
    frame = pd.DataFrame(
        {
            "name": pd.array(["a", "b", None, "a", "b", "a"], dtype="string"),
            "count": np.array([1, 2, 3, 4, 5, 6], dtype="int64"),
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.5],
            "w": np.array([1, 2, 3, 4, 5, 6], dtype="float32"),
        }
    )
    result = impute_mice(frame, max_iter=2, n_imputations=2, return_result=True)
    for completed in result.data.imputations:
        assert list(completed.columns) == list(frame.columns)
        assert completed["count"].dtype == "int64"
        assert completed["w"].dtype == np.float32
        assert completed["name"].dtype == "string"


def test_timedelta_and_datetime_predictors_are_neither_modelled_nor_altered() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "gap": pd.to_timedelta([1, 2, 3, 4, 5, 6], unit="D"),
            "when": pd.to_datetime(["2024-01-0%d" % day for day in range(1, 7)]),
        }
    )
    for result in (
        impute_knn(frame, n_neighbors=2),
        impute_rf(frame, n_estimators=3),
        impute_mice(frame, max_iter=2),
    ):
        pd.testing.assert_series_equal(result["gap"], frame["gap"])
        pd.testing.assert_series_equal(result["when"], frame["when"])
        assert not result["x"].isna().any()


def test_mean_imputation_leaves_complete_columns_untouched_and_mode_fills_flags() -> (
    None
):
    frame = pd.DataFrame(
        {
            "count": np.array([1, 2, 3, 4], dtype="int64"),
            "x": [1.0, np.nan, 3.0, 5.0],
            "flag": pd.array([True, None, True, False], dtype="boolean"),
            "when": pd.to_datetime(["2024-01-01", None, "2024-01-01", "2024-01-02"]),
        }
    )
    result = impute_mean(frame)
    pd.testing.assert_series_equal(result["count"], frame["count"])
    assert result["x"].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert result["flag"].tolist() == [True, True, True, False]
    assert result["flag"].dtype == "boolean"
    assert result["when"].dtype == frame["when"].dtype
    assert result["when"].iloc[1] == pd.Timestamp("2024-01-01")


def test_numeric_only_mean_ignores_an_all_missing_non_numeric_column() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0], "label": pd.Series([None, None, None], dtype=object)}
    )
    result = impute_mean(frame, numeric_only=True)
    assert result["x"].tolist() == [1.0, 2.0, 3.0]
    assert result["label"].isna().all()


def test_require_observed_names_the_all_missing_column() -> None:
    _require_observed(pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan]}), ["a"])
    with pytest.raises(Exception, match="'b'"):
        _require_observed(pd.DataFrame({"a": [1.0], "b": [np.nan]}))


def test_missing_to_nan_handles_object_and_non_object_columns() -> None:
    plain = np.array([1.0, np.nan])
    assert _missing_to_nan(plain) is plain
    complete = np.array(["a", "b"], dtype=object)
    assert _missing_to_nan(complete) is complete
    cleaned = _missing_to_nan(np.array([1, pd.NA, None], dtype=object))
    assert cleaned.tolist()[0] == 1
    assert np.isnan(cleaned[1]) and np.isnan(cleaned[2])


def test_fill_categorical_gaps_uses_the_reference_mode_only_where_available() -> None:
    result = pd.DataFrame({"g": ["a", np.nan], "h": [np.nan, np.nan]})
    reference = pd.DataFrame({"g": ["b", "b"], "h": [np.nan, np.nan]})
    filled = _fill_categorical_gaps(result, ["g", "h"], reference)
    assert filled["g"].tolist() == ["a", "b"]
    assert filled["h"].isna().all()


def test_assign_labels_casts_to_extension_dtypes_and_passes_numpy_through() -> None:
    frame = pd.DataFrame(
        {
            "n": pd.array([1, None], dtype="Int64"),
            "f": [1.0, np.nan],
        }
    )
    mask = frame["n"].isna()
    _assign_labels(frame, mask, "n", np.array([7.0]))
    _assign_labels(frame, mask, "f", np.array([2.0]))
    assert frame["n"].tolist() == [1, 7]
    assert frame["f"].tolist() == [1.0, 2.0]
