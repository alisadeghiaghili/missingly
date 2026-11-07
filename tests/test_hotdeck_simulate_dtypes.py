"""Dtype contracts for hot-deck imputation and missingness simulation (WP-04).

Findings covered: F-012, F-232, F-233, F-358.

* The characterization tests pin the output of plain float64 + object frames
  with a fixed seed. They must keep passing unchanged.
* Hot-deck donors copy the donor's actual value: boolean columns keep their
  dtype, Int64 keeps exact integers and tz-aware datetimes keep their zone.
* Masking a bool column in ``missingly.simulate`` stores ``pd.NA`` in a nullable
  ``boolean`` column and must not emit a FutureWarning.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from missingly.hotdeck import (
    impute_hotdeck_random,
    impute_hotdeck_sequential,
    impute_hotdeck_weighted,
)
from missingly.simulate import (
    simulate_mar,
    simulate_mcar,
    simulate_mixed,
    simulate_mnar,
)

STRICT_FUTURE_WARNINGS = pytest.mark.filterwarnings("error::FutureWarning")
TZ = "Europe/Berlin"


def _hotdeck_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0],
            "y": [10.0, 20.0, np.nan, 40.0, 50.0, np.nan, 70.0, 80.0],
            "cat": ["a", None, "b", "a", "b", "a", None, "b"],
        }
    )


def _simulation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [float(v) for v in range(1, 13)],
            "y": [1.5 * v for v in range(12)],
            "cat": list("abcdabcdabcd"),
        }
    )


def _bool_frame() -> pd.DataFrame:
    return pd.DataFrame({"flag": [True, False] * 6, "p": list(range(12))})


def _tz_frame() -> pd.DataFrame:
    stamps = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-02", None, "2020-01-04"], utc=True)
    )
    return pd.DataFrame({"t": stamps.dt.tz_convert(TZ)})


def _tz_complete_frame() -> pd.DataFrame:
    stamps = pd.Series(
        pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], utc=True
        )
    )
    return pd.DataFrame({"t": stamps.dt.tz_convert(TZ)})


# ---------------------------------------------------------------------------
# Characterization: float64 + object frames, fixed seed. MCAR, hot-deck and
# dtypes must not change; the MAR / MNAR masks were re-pinned when the two
# mechanisms were calibrated to the requested frac (WP-19).
# ---------------------------------------------------------------------------


def test_characterization_random_hotdeck_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [1.0, 8.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0],
            "y": [10.0, 20.0, 50.0, 40.0, 50.0, 20.0, 70.0, 80.0],
            "cat": ["a", "b", "b", "a", "b", "a", "a", "b"],
        }
    )
    result = impute_hotdeck_random(_hotdeck_frame(), random_state=0)
    assert_frame_equal(result, expected)


def test_characterization_sequential_hotdeck_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [1.0, 1.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0],
            "y": [10.0, 20.0, 20.0, 40.0, 50.0, 50.0, 70.0, 80.0],
            "cat": ["a", "a", "b", "a", "b", "a", "a", "b"],
        }
    )
    result = impute_hotdeck_sequential(_hotdeck_frame())
    assert_frame_equal(result, expected)


def test_characterization_weighted_hotdeck_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [1.0, 1.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0],
            "y": [10.0, 20.0, 50.0, 40.0, 50.0, 40.0, 70.0, 80.0],
            "cat": ["a", "a", "b", "a", "b", "a", "a", "b"],
        }
    )
    result = impute_hotdeck_weighted(_hotdeck_frame(), n_donors=2, random_state=0)
    assert_frame_equal(result, expected)


def test_characterization_simulate_mcar_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                np.nan,
                np.nan,
                np.nan,
                10.0,
                11.0,
                12.0,
            ],
            "y": [
                np.nan,
                1.5,
                3.0,
                4.5,
                6.0,
                7.5,
                9.0,
                10.5,
                12.0,
                13.5,
                np.nan,
                np.nan,
            ],
            "cat": list("abcdabcdabcd"),
        }
    )
    result = simulate_mcar(_simulation_frame(), 0.25, random_state=0)
    assert_frame_equal(result, expected)


def test_characterization_simulate_mar_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [float(v) for v in range(1, 13)],
            "y": [
                0.0,
                1.5,
                np.nan,
                np.nan,
                6.0,
                7.5,
                np.nan,
                10.5,
                np.nan,
                13.5,
                15.0,
                np.nan,
            ],
            "cat": list("abcdabcdabcd"),
        }
    )
    result = simulate_mar(
        _simulation_frame(), 0.4, target_col="y", predictor_col="x", random_state=0
    )
    assert_frame_equal(result, expected)


def test_characterization_simulate_mnar_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [float(v) for v in range(1, 13)],
            "y": [
                0.0,
                1.5,
                np.nan,
                np.nan,
                6.0,
                7.5,
                np.nan,
                10.5,
                np.nan,
                13.5,
                15.0,
                np.nan,
            ],
            "cat": list("abcdabcdabcd"),
        }
    )
    result = simulate_mnar(_simulation_frame(), 0.4, target_col="y", random_state=0)
    assert_frame_equal(result, expected)


def test_characterization_simulate_mixed_float_object_frame():
    expected = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                10.0,
                11.0,
                np.nan,
            ],
            "y": [np.nan, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, np.nan, 16.5],
            "cat": list("abcdabcdabcd"),
        }
    )
    mechanisms = [
        {"mechanism": "mcar", "frac": 0.2},
        {"mechanism": "mnar", "frac": 0.3, "target_col": "x"},
    ]
    result = simulate_mixed(_simulation_frame(), mechanisms, random_state=1)
    assert_frame_equal(result, expected)


def test_simulate_mcar_upcasts_int64_to_float64_as_before():
    # Int64 handling is unchanged: masking an int64 column still yields float64.
    df = pd.DataFrame({"n": np.arange(10, dtype="int64")})
    result = simulate_mcar(df, 0.3, columns=["n"], random_state=0)
    assert result["n"].dtype == np.float64
    assert int(result["n"].isna().sum()) == 3


# ---------------------------------------------------------------------------
# F-012: weighted hot-deck must accept bool and boolean columns.
# ---------------------------------------------------------------------------


def test_weighted_hotdeck_accepts_complete_numpy_bool_predictor():
    df = pd.DataFrame(
        {
            "flag": [True, False, True, False, True],
            "x": [1.0, np.nan, 3.0, 4.0, 5.0],
        }
    )
    result = impute_hotdeck_weighted(df, n_donors=2, random_state=0)
    assert result["flag"].dtype == bool
    assert result["x"].notna().all()
    assert result["x"].iloc[1] in {1.0, 3.0, 4.0, 5.0}


def test_weighted_hotdeck_keeps_boolean_dtype_of_imputed_column():
    df = pd.DataFrame(
        {
            "b": pd.array([True, None, False, True, False], dtype="boolean"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = impute_hotdeck_weighted(df, n_donors=2, random_state=0)
    assert result["b"].dtype == "boolean"
    assert result["b"].notna().all()
    assert result["b"].iloc[1] in (True, False)


# ---------------------------------------------------------------------------
# F-233: donor values must not lose precision through a coerced row Series.
# ---------------------------------------------------------------------------


def test_weighted_hotdeck_keeps_exact_int64_values_for_large_integers():
    big = 2**53 + 1
    df = pd.DataFrame(
        {
            "i": pd.array([big, None, big + 2, 5], dtype="Int64"),
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = impute_hotdeck_weighted(df, n_donors=1, random_state=0)
    assert result["i"].dtype == "Int64"
    assert int(result["i"].iloc[1]) in (big, big + 2)


# ---------------------------------------------------------------------------
# F-232: tz-aware datetimes must not go through a naive numpy round trip.
# ---------------------------------------------------------------------------


@STRICT_FUTURE_WARNINGS
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_random_hotdeck_keeps_timezone_of_datetime_column():
    df = _tz_frame()
    result = impute_hotdeck_random(df, random_state=0)
    assert str(result["t"].dt.tz) == TZ
    assert result["t"].notna().all()
    assert result["t"].iloc[2] in set(df["t"].dropna())


@STRICT_FUTURE_WARNINGS
def test_sequential_hotdeck_keeps_timezone_of_datetime_column():
    result = impute_hotdeck_sequential(_tz_frame())
    assert str(result["t"].dt.tz) == TZ
    assert result["t"].notna().all()


@STRICT_FUTURE_WARNINGS
def test_simulate_mcar_on_tz_aware_column_emits_no_future_warning():
    result = simulate_mcar(_tz_complete_frame(), 0.5, columns=["t"], random_state=0)
    assert str(result["t"].dt.tz) == TZ


# ---------------------------------------------------------------------------
# F-358: masking a bool column must not raise or warn; it becomes boolean.
# ---------------------------------------------------------------------------


@STRICT_FUTURE_WARNINGS
def test_simulate_mcar_masks_bool_column_as_nullable_boolean():
    result = simulate_mcar(_bool_frame(), 0.25, columns=["flag"], random_state=0)
    assert result["flag"].dtype == "boolean"
    assert int(result["flag"].isna().sum()) == 3


@STRICT_FUTURE_WARNINGS
def test_simulate_mar_masks_bool_target_as_nullable_boolean():
    df = _bool_frame()
    result = simulate_mar(df, 0.4, target_col="flag", predictor_col="p", random_state=0)
    assert result["flag"].dtype == "boolean"
    assert result["flag"].isna().any()
    kept = result["flag"].notna()
    assert (result.loc[kept, "flag"].astype(bool) == df.loc[kept, "flag"]).all()


@STRICT_FUTURE_WARNINGS
def test_simulate_mnar_masks_bool_target_as_nullable_boolean():
    df = _bool_frame()
    result = simulate_mnar(df, 0.4, target_col="flag", random_state=0)
    assert result["flag"].dtype == "boolean"
    assert result["flag"].isna().any()


@STRICT_FUTURE_WARNINGS
def test_simulate_mixed_masks_bool_column_as_nullable_boolean():
    mechanisms = [{"mechanism": "mcar", "frac": 0.25, "columns": ["flag"]}]
    result = simulate_mixed(_bool_frame(), mechanisms, random_state=0)
    assert result["flag"].dtype == "boolean"
    assert int(result["flag"].isna().sum()) == 3
