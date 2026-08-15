"""Behavioral contracts shared by visualization base and Plotly backends."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from missingly.visualisation import _base, interactive


def test_nullity_handles_mixed_sentinels_without_mutating_input() -> None:
    """Sentinel-aware nullity must preserve the mixed-dtype caller frame."""
    frame = pd.DataFrame(
        {
            "number": [1, -99, 3],
            "label": ["ok", "N/A", None],
            "flag": pd.array([True, pd.NA, False], dtype="boolean"),
        }
    )
    original = frame.copy(deep=True)

    result = _base._nullity(frame, missing_values=[-99, "N/A"])

    assert result.to_dict("list") == {
        "number": [False, True, False],
        "label": [False, True, True],
        "flag": [False, True, False],
    }
    pd.testing.assert_frame_equal(frame, original)


def test_plotly_requirement_is_lazy_and_reports_the_optional_install(
    monkeypatch,
) -> None:
    """Missing Plotly must fail only at the interactive boundary with guidance."""
    original_import = builtins.__import__

    def import_without_plotly(name, globals=None, locals=None, fromlist=(), level=0):
        """Reject only Plotly imports while leaving unrelated imports intact."""
        if name == "plotly.graph_objects":
            raise ImportError("Plotly intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_plotly)

    with pytest.raises(ImportError, match=r"missingly\[interactive\]"):
        _base._require_plotly()


def test_plotly_vis_miss_handles_empty_rows_when_clustering_is_requested() -> None:
    """An empty-column frame has an annotated Plotly result instead of SciPy failure."""
    frame = pd.DataFrame(index=["first", "second"])

    with pytest.warns(UserWarning, match="original order"):
        figure = interactive._vis_miss_plotly(frame, cluster=True)

    assert figure.data[0].z.shape == (2, 0)
    assert list(figure.data[0].y) == ["first", "second"]


def test_plotly_vis_miss_does_not_hide_unexpected_clustering_errors(
    monkeypatch,
) -> None:
    """Unexpected SciPy failures must remain observable rather than become output."""
    def fail_linkage(*args, **kwargs):
        """Reproduce an unsupported clustering backend failure."""
        raise RuntimeError("unexpected linkage backend failure")

    monkeypatch.setattr("scipy.cluster.hierarchy.linkage", fail_linkage)
    frame = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})

    with pytest.raises(RuntimeError, match="unexpected linkage backend failure"):
        interactive._vis_miss_plotly(frame, cluster=True)


def test_plotly_upset_empty_data_returns_an_annotated_figure() -> None:
    """An all-observed frame has a deterministic, inspectable empty UpSet result."""
    figure = interactive._upset_plotly(pd.DataFrame({"a": [1.0, 2.0]}))

    assert not figure.data
    assert figure.layout.annotations[0].text == "No missing values to plot."
