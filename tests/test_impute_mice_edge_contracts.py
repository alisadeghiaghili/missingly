"""Public edge-case contracts for :func:`missingly.impute.impute_mice`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import InsufficientDataError
from missingly.impute import _fill_feature_matrix, impute_mice


def test_fill_feature_matrix_handles_all_nan_features_without_warning() -> None:
    """All-NaN features use the documented zero fallback without RuntimeWarning."""
    observed = np.array([[np.nan, 2.0], [np.nan, 4.0]])
    incomplete = np.array([[np.nan, 6.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        observed_filled, incomplete_filled = _fill_feature_matrix(
            observed,
            incomplete,
        )

    np.testing.assert_array_equal(observed_filled, [[0.0, 2.0], [0.0, 4.0]])
    np.testing.assert_array_equal(incomplete_filled, [[0.0, 6.0]])
    np.testing.assert_array_equal(observed, [[np.nan, 2.0], [np.nan, 4.0]])
    np.testing.assert_array_equal(incomplete, [[np.nan, 6.0]])


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pd.DataFrame({"unknown": [np.nan, np.nan], "known": [1.0, 2.0]}),
            id="all-missing-column",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "unknown": [np.nan],
                    "label": pd.array(["known"], dtype="string"),
                }
            ),
            id="single-row-without-donor",
        ),
    ],
)
def test_impute_mice_rejects_columns_without_observed_donors(
    frame: pd.DataFrame,
) -> None:
    """Columns with no donors fail with the documented error and no mutation."""
    original = frame.copy(deep=True)

    with pytest.raises(InsufficientDataError, match="unknown"):
        impute_mice(frame, max_iter=2, random_state=7)

    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        pytest.param(
            {"visit_sequence": ("unknown",)},
            TypeError,
            "visit_sequence",
            id="visit-sequence-type",
        ),
        pytest.param(
            {"predictor_matrix": object()},
            TypeError,
            "predictor_matrix",
            id="predictor-matrix-type",
        ),
        pytest.param(
            {"bounds": []},
            TypeError,
            "bounds",
            id="bounds-type",
        ),
    ],
)
def test_impute_mice_validates_options_before_no_donor_preflight(
    options: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid configuration has priority over the data-dependent donor error."""
    frame = pd.DataFrame({"unknown": [np.nan, np.nan], "known": [1.0, 2.0]})
    original = frame.copy(deep=True)

    with pytest.raises(error_type, match=message):
        impute_mice(frame, max_iter=2, random_state=7, **options)

    pd.testing.assert_frame_equal(frame, original)


def test_impute_mice_handles_nullable_strings_deterministically() -> None:
    """Nullable strings encode safely, retain their dtype, and preserve observations."""
    frame = pd.DataFrame(
        {
            "score": pd.array([1.0, None, 3.0, 4.0], dtype="Float64"),
            "label": pd.array(["low", None, "high", "low"], dtype="string"),
        }
    )
    original = frame.copy(deep=True)

    first = impute_mice(frame, max_iter=2, random_state=17)
    second = impute_mice(frame, max_iter=2, random_state=17)

    pd.testing.assert_frame_equal(first, second)
    assert first["label"].dtype == frame["label"].dtype
    assert first.isna().sum().sum() == 0
    assert first.loc[0, "score"] == frame.loc[0, "score"]
    assert first.loc[2, "label"] == frame.loc[2, "label"]
    pd.testing.assert_frame_equal(frame, original)
