"""Public edge-case contracts for :func:`missingly.impute.impute_mice`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import InsufficientDataError
from missingly.impute import impute_mice


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
