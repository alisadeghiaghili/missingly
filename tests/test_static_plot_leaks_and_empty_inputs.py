"""Figure-lifetime and degenerate-input contracts for the static plots.

Functions that create their own figure must release it when they fail after
creating it (F-375). Functions called with a caller's *ax* must never create a
figure (F-366). Degenerate frames (no rows, no columns, no missing values,
only missing values) must either raise a clear ``ValueError`` or return an
annotated empty-state plot without runtime warnings (F-302, F-367, F-374,
F-380).
"""

from __future__ import annotations

import contextlib
import inspect
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from missingly.visualisation import static

ZERO_ROWS = "zero_rows"
ZERO_COLS = "zero_cols"
NO_MISSING = "no_missing"
ALL_MISSING = "all_missing"

PUBLIC_FUNCTIONS = [
    "matrix",
    "bar",
    "miss_case",
    "miss_var_pct",
    "miss_which",
    "upset",
    "miss_patterns",
    "miss_cooccurrence",
    "heatmap",
    "dendrogram",
    "vis_miss",
    "miss_cluster",
    "miss_row_profile",
    "shadow_scatter",
    "scatter_miss",
    "vis_miss_fct",
    "vis_miss_by_group",
    "vis_impute_dist",
    "miss_impute_compare",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
    "vis_parallel_coords",
]

# Public functions that take a caller-supplied axes through ``ax``.
AX_FUNCTIONS = [
    name
    for name in PUBLIC_FUNCTIONS
    if "ax" in inspect.signature(getattr(static, name)).parameters
]

# Extra positional arguments after the frame (column names).
_EXTRA_ARGS: dict[str, tuple[str, ...]] = {
    "shadow_scatter": ("a", "b"),
    "scatter_miss": ("a", "b"),
    "vis_miss_fct": ("g",),
    "vis_miss_by_group": ("g",),
    "vis_impute_dist": ("a",),
    "vis_miss_span": ("a",),
}
_TAKES_IMPUTED = frozenset({"vis_impute_dist", "miss_impute_compare"})

_GENERIC_TEXT = {
    ZERO_ROWS: "No rows to display",
    ZERO_COLS: "No columns to display",
    NO_MISSING: "No missing values",
    ALL_MISSING: "All values are missing",
}

# Per-function outcomes that differ from the generic empty state.
# ("raise", message) or ("text", message).
_OVERRIDES: dict[str, dict[str, tuple[str, str]]] = {
    "matrix": {
        ZERO_ROWS: ("raise", "matrix() needs at least one row"),
        ZERO_COLS: ("raise", "matrix() needs at least one column"),
    },
    "miss_cooccurrence": {
        ZERO_COLS: ("raise", "miss_cooccurrence() needs at least one column"),
    },
    "vis_miss_fct": {
        ZERO_ROWS: ("raise", "vis_miss_fct() needs at least one row"),
        ZERO_COLS: ("raise", "vis_miss_fct() needs at least one column"),
    },
    "shadow_scatter": {
        ZERO_COLS: ("raise", "shadow_scatter() needs at least one column"),
    },
    "scatter_miss": {
        ZERO_COLS: ("raise", "shadow_scatter() needs at least one column"),
    },
    "vis_impute_dist": {
        ZERO_COLS: ("raise", "vis_impute_dist() needs at least one column"),
    },
    "vis_miss_by_group": {
        ZERO_COLS: ("raise", "vis_miss_by_group() needs at least one column"),
    },
    "vis_miss_span": {
        ZERO_COLS: ("raise", "vis_miss_span() needs at least one column"),
    },
    "heatmap": {
        ZERO_ROWS: ("text", "Not enough columns with missing values"),
        ZERO_COLS: ("text", "Not enough columns with missing values"),
        NO_MISSING: ("text", "Not enough columns with missing values"),
    },
    "dendrogram": {
        ZERO_ROWS: ("text", "Not enough columns with missing values"),
        ZERO_COLS: ("text", "Not enough columns with missing values"),
        NO_MISSING: ("text", "Not enough columns with missing values"),
    },
    "miss_row_profile": {
        ZERO_ROWS: ("text", "No row completeness data to display"),
        ZERO_COLS: ("text", "No row completeness data to display"),
    },
    "vis_parallel_coords": {
        ZERO_ROWS: ("text", "No numeric columns available"),
        ZERO_COLS: ("text", "No numeric columns available"),
    },
    "upset": {
        ZERO_ROWS: ("text", "No missingness patterns"),
        ZERO_COLS: ("text", "No missingness patterns"),
        NO_MISSING: ("text", "No missingness patterns"),
    },
    "miss_impute_compare": {
        ZERO_ROWS: ("raise", "No numeric columns with missing values"),
        ZERO_COLS: ("raise", "No numeric columns with missing values"),
        NO_MISSING: ("raise", "No numeric columns with missing values"),
    },
}


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close every pyplot figure before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [np.nan, 2.0, 3.0, 5.0],
            "g": ["x", "y", "x", "y"],
        }
    )


def _degenerate_frames() -> dict[str, pd.DataFrame]:
    return {
        ZERO_ROWS: pd.DataFrame(
            {
                "a": pd.Series([], dtype=float),
                "b": pd.Series([], dtype=float),
                "g": pd.Series([], dtype=object),
            }
        ),
        ZERO_COLS: pd.DataFrame(index=range(3)),
        NO_MISSING: pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "g": ["x", "y", "x"]}
        ),
        ALL_MISSING: pd.DataFrame(
            {
                "a": [np.nan] * 3,
                "b": [np.nan] * 3,
                "g": pd.Series([np.nan] * 3, dtype=object),
            }
        ),
    }


def _invoke(name: str, frame: pd.DataFrame, **kwargs: Any) -> Any:
    func = getattr(static, name)
    extra = _EXTRA_ARGS.get(name, ())
    if name in _TAKES_IMPUTED:
        return func(frame, frame.fillna(0.5), *extra, **kwargs)
    return func(frame, *extra, **kwargs)


def _texts(result: Any) -> list[str]:
    if isinstance(result, dict):
        axes = list(result.values())
    elif isinstance(result, Figure):
        axes = list(result.axes)
    else:
        axes = [result]
    return [text.get_text() for axis in axes for text in axis.texts]


def _expected(name: str, case: str) -> tuple[str, str]:
    return _OVERRIDES.get(name, {}).get(case, ("text", _GENERIC_TEXT[case]))


def test_every_public_static_function_is_covered() -> None:
    public = sorted(
        name
        for name, value in vars(static).items()
        if inspect.isfunction(value)
        and value.__module__ == static.__name__
        and not name.startswith("_")
    )
    assert public == sorted(PUBLIC_FUNCTIONS)


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_failure_after_figure_creation_releases_the_figure(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A figure created by the call is closed when the call raises (F-375)."""
    before = plt.get_fignums()

    def fail_after_creation(self: Axes, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom after figure creation")

    monkeypatch.setattr(Axes, "__init__", fail_after_creation)
    with pytest.raises(RuntimeError, match="boom after figure creation"):
        _invoke(name, _valid_frame())
    assert plt.get_fignums() == before


@pytest.mark.parametrize("call", ["shadow_scatter", "vis_impute_dist"])
def test_bad_column_name_does_not_leave_an_orphan_figure(call: str) -> None:
    """A KeyError for a misspelled column must not leave a figure open (F-375)."""
    frame = _valid_frame()
    before = plt.get_fignums()
    with pytest.raises(KeyError):
        if call == "shadow_scatter":
            static.shadow_scatter(frame, "missing_column", "b")
        else:
            static.vis_impute_dist(frame, frame.fillna(0.5), "missing_column")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("name", AX_FUNCTIONS)
def test_caller_axes_never_create_a_new_figure(name: str) -> None:
    """Passing *ax* must draw on it and create no figure (F-366)."""
    _, ax = plt.subplots()
    before = plt.get_fignums()
    _invoke(name, _valid_frame(), ax=ax)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("case", sorted(_degenerate_frames()))
@pytest.mark.parametrize("name", AX_FUNCTIONS)
def test_caller_axes_stay_on_degenerate_input(name: str, case: str) -> None:
    """Empty-state and error paths must also respect the caller's *ax*."""
    _, ax = plt.subplots()
    before = plt.get_fignums()
    with contextlib.suppress(ValueError):
        _invoke(name, _degenerate_frames()[case], ax=ax)
    assert plt.get_fignums() == before


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("case", sorted(_degenerate_frames()))
@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_degenerate_input_matches_the_decided_behaviour(name: str, case: str) -> None:
    """Each public function x degenerate case raises or shows an empty state."""
    frame = _degenerate_frames()[case]
    kind, message = _expected(name, case)
    if kind == "raise":
        with pytest.raises(ValueError, match=re.escape(message)):
            _invoke(name, frame)
        return
    result = _invoke(name, frame)
    assert result is not None
    assert any(message in text for text in _texts(result))
