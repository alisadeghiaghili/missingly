"""Public empty-input contracts for row missingness visualisation."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from missingly.visualisation.static import miss_row_profile


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(index=["first", "second"]),
        pd.DataFrame({"score": pd.Series(dtype="float64")}),
    ],
    ids=["no-columns", "no-rows"],
)
def test_miss_row_profile_annotates_empty_axes_without_runtime_warnings(
    frame: pd.DataFrame,
) -> None:
    """Empty inputs return an explanatory axes rather than histogram failures."""
    figure, axes = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = miss_row_profile(frame, ax=axes)

    assert result is axes
    assert result.figure is figure
    assert result.texts
    assert "No row completeness data" in result.texts[0].get_text()
