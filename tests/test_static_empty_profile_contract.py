"""Public empty-input contracts for row missingness visualisation."""

from __future__ import annotations

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

    with warnings_as_errors():
        result = miss_row_profile(frame, ax=axes)

    assert result is axes
    assert result.figure is figure
    assert result.texts
    assert "No row completeness data" in result.texts[0].get_text()


class warnings_as_errors:
    """Turn unexpected runtime warnings into test failures for one plot call."""

    def __enter__(self) -> None:
        import warnings

        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def __exit__(self, *args: object) -> None:
        self._context.__exit__(*args)
