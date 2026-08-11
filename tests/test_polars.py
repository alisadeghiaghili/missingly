"""Contract tests for native Polars missingness summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from missingly.polars_adapter import polars_missing_summary

pl = pytest.importorskip("polars")


def test_polars_summary_preserves_eager_polars_type_and_null_policy() -> None:
    """An eager Polars frame gets a Polars summary without pandas conversion."""
    frame = pl.DataFrame({"age": [20, None, 40], "city": [None, "A", "B"]})

    result = polars_missing_summary(frame)

    assert isinstance(result, pl.DataFrame)
    assert result.to_dicts() == [
        {"variable": "age", "n_missing": 1, "n_complete": 2, "pct_missing": 100 / 3},
        {"variable": "city", "n_missing": 1, "n_complete": 2, "pct_missing": 100 / 3},
    ]


def test_polars_summary_preserves_lazy_execution() -> None:
    """A LazyFrame yields a LazyFrame and only materializes when requested."""
    frame = pl.LazyFrame({"x": [1, None, 3], "y": [None, None, 2]})

    result = polars_missing_summary(frame)

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().to_dicts() == [
        {"variable": "x", "n_missing": 1, "n_complete": 2, "pct_missing": 100 / 3},
        {"variable": "y", "n_missing": 2, "n_complete": 1, "pct_missing": 200 / 3},
    ]


def test_polars_summary_rejects_non_polars_inputs() -> None:
    """The native API rejects pandas rather than silently copying its data."""
    with pytest.raises(TypeError, match="polars DataFrame or LazyFrame"):
        polars_missing_summary(pd.DataFrame({"x": [1.0, None]}))
