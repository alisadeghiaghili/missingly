"""Regression tests for the individual nullable-dtype audit findings (WP-04).

Each test is named after the finding it reproduces.  F-051, F-055 and F-058
were reported without an independent reproduction; the comments record what
each one turned out to be on the supported toolchain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.impute import (
    _encode_features,
    impute_gb,
    impute_mice,
    impute_pmm,
    impute_rf,
)

pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _nullable_frame(dtype: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": pd.array([1, 2, None, 4, 5, 7, 8, 9], dtype=dtype),
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, np.nan],
        }
    )


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
@pytest.mark.parametrize("imputer", [impute_rf, impute_gb])
def test_f051_f058_f093_forest_imputers_accept_nullable_numeric_columns(
    imputer, dtype
) -> None:
    # Reproduced: IndexError from ``X_full[observed_mask]`` for Int64 and Float64.
    result = imputer(_nullable_frame(dtype), n_estimators=3, random_state=0)
    assert not result.isna().any().any()
    assert result["x"].dtype == _nullable_frame(dtype)["x"].dtype


@pytest.mark.parametrize("imputer", [impute_rf, impute_gb])
def test_f058_boolean_column_is_not_affected(imputer) -> None:
    # Not reproduced for the nullable ``boolean`` dtype: it takes the
    # categorical branch.  The claim in F-058 is therefore limited to the
    # nullable numeric dtypes; this test pins the working behaviour.
    frame = pd.DataFrame(
        {
            "y": np.arange(8.0),
            "b": pd.array(
                [True, False, None, True, False, True, False, True], dtype="boolean"
            ),
        }
    )
    result = imputer(frame, n_estimators=3, random_state=0)
    assert result["b"].dtype == "boolean"
    assert not result["b"].isna().any()


def test_f051_f020_mice_history_accepts_nullable_integers() -> None:
    # Reproduced: TypeError (float() of pd.NA) in ``_run_mice_with_history``.
    frame = _nullable_frame("Int64")
    result, history = impute_mice(
        frame, max_iter=2, random_state=0, return_history=True
    )
    assert result["x"].dtype == "Int64"
    assert not result.isna().any().any()
    assert set(history) == {"x", "y"}


def test_f055_pmm_accepts_nullable_predictors_with_missing_values() -> None:
    # Reproduced with two or more predictors where one is a nullable column that
    # still holds pd.NA when the target is processed (``.values`` becomes an
    # object array).  A single nullable predictor happens to work.
    frame = pd.DataFrame(
        {
            "y": [2.0, 4.0, 6.0, np.nan, 10.0, 12.0, 14.0, 16.0],
            "x": pd.array([1, 2, None, 4, 5, 7, 8, 9], dtype="Int64"),
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    result = impute_pmm(frame, random_state=0)
    assert not result.isna().any().any()
    assert result["x"].dtype == "Int64"


def test_f014_nullable_numeric_features_keep_their_magnitude_order() -> None:
    # Int64 columns with pd.NA reach the logistic imputers as object arrays.
    values = pd.DataFrame(
        {
            "n": pd.array([9, 100, 20, 10, None, 3], dtype="Int64"),
            "f": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    assert values.to_numpy().dtype == object
    encoded = _encode_features(values.to_numpy())
    observed = encoded[:, 0][~np.isnan(encoded[:, 0])]
    np.testing.assert_array_equal(observed, [9.0, 100.0, 20.0, 10.0, 3.0])
    assert np.isnan(encoded[4, 0])
