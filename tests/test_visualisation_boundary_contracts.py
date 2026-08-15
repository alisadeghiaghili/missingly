"""Behavioral contracts shared by visualization base and Plotly backends."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from missingly.visualisation import _base, interactive, static


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


def test_public_plotly_vis_miss_preserves_empty_columns_when_clustered() -> None:
    """The public Plotly heatmap preserves empty-column rows after clustering fails."""
    frame = pd.DataFrame(index=["first", "second"])

    with pytest.warns(UserWarning, match="original order"):
        figure = static.vis_miss(frame, interactive=True, cluster=True)

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
        static.vis_miss(frame, interactive=True, cluster=True)


def test_public_plotly_vis_miss_honors_column_sorting() -> None:
    """The public Plotly missingness plot sorts columns by missingness."""
    frame = pd.DataFrame(
        {
            "lowest": [1.0, None, 1.0, 1.0],
            "highest": [None, None, None, 1.0],
            "middle": [1.0, None, 1.0, 1.0],
        },
        index=["first", "second", "third", "fourth"],
    )

    missingness = static.vis_miss(frame, interactive=True, sort=True)

    assert list(missingness.data[0].x) == [
        "highest (75.0%)",
        "lowest (25.0%)",
        "middle (25.0%)",
    ]


def test_public_plotly_miss_var_pct_honors_column_sorting() -> None:
    """The public Plotly variable percentages retain the requested order."""
    frame = pd.DataFrame({"first": [None, 1.0], "second": [1.0, 1.0]})

    percentages = static.miss_var_pct(frame, interactive=True, sort=False)

    assert list(percentages.data[0].y) == ["first", "second"]


def test_public_plotly_bar_honors_column_sorting() -> None:
    """The public Plotly column bar chart sorts missing counts when requested."""
    frame = pd.DataFrame(
        {"lowest": [1.0, None], "highest": [None, None], "middle": [1.0, None]}
    )

    variables = static.bar(frame, interactive=True, sort=True)

    assert list(variables.data[0].x) == ["highest", "lowest", "middle"]


def test_public_plotly_miss_case_honors_row_sorting() -> None:
    """The public Plotly case bar chart sorts rows by missing counts."""
    frame = pd.DataFrame(
        {
            "left": [1.0, None, 1.0, 1.0],
            "right": [1.0, None, None, 1.0],
        },
        index=["first", "second", "third", "fourth"],
    )

    cases = static.miss_case(frame, interactive=True, sort=True)

    assert list(cases.data[0].x) == ["second", "third", "first", "fourth"]


def test_public_plotly_upset_honors_minimum_subset_size() -> None:
    """The public Plotly UpSet chart removes combinations below its threshold."""
    frame = pd.DataFrame(
        {
            "lowest": [1.0, None, 1.0, 1.0],
            "highest": [None, None, None, 1.0],
            "middle": [1.0, None, 1.0, 1.0],
        }
    )

    patterns = static.upset(frame, interactive=True, min_subset_size=2)

    assert list(patterns.data[0].y) == [2]


@pytest.mark.parametrize(
    ("function_name", "interactive"),
    [
        pytest.param("vis_miss", False, id="static-vis-miss"),
        pytest.param("miss_var_pct", False, id="static-miss-var-pct"),
        pytest.param("bar", False, id="static-bar"),
        pytest.param("miss_case", False, id="static-miss-case"),
        pytest.param("vis_miss", True, id="plotly-vis-miss"),
        pytest.param("miss_var_pct", True, id="plotly-miss-var-pct"),
        pytest.param("bar", True, id="plotly-bar"),
        pytest.param("miss_case", True, id="plotly-miss-case"),
    ],
)
def test_public_sorting_requests_stable_tie_order(
    monkeypatch,
    function_name: str,
    interactive: bool,
) -> None:
    """Public static and Plotly sort options explicitly preserve caller tie order."""
    original_sort_values = pd.Series.sort_values

    def require_stable_sort(series, *args, **kwargs):
        """Require visualization sorting to opt into the deterministic algorithm."""
        assert kwargs.get("kind") == "stable"
        return original_sort_values(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "sort_values", require_stable_sort)
    frame = pd.DataFrame(
        {"first": [None, 1.0], "second": [1.0, None]},
        index=["row-first", "row-second"],
    )

    plot = getattr(static, function_name)
    plot(frame, sort=True, interactive=interactive)


def test_public_bar_reports_missing_counts_on_both_backends() -> None:
    """Static and Plotly bar charts expose the documented column counts."""
    frame = pd.DataFrame({"two": [None, None, 1.0], "one": [None, 1.0, 1.0]})

    static_axis = static.bar(frame)
    interactive_figure = static.bar(frame, interactive=True)

    assert [patch.get_height() for patch in static_axis.patches] == [2.0, 1.0]
    assert static_axis.get_ylabel() == "Number of Missing Values"
    assert list(interactive_figure.data[0].y) == [2, 1]


def test_plotly_upset_empty_data_returns_an_annotated_figure() -> None:
    """An all-observed frame has a deterministic, inspectable empty UpSet result."""
    figure = interactive._upset_plotly(pd.DataFrame({"a": [1.0, 2.0]}))

    assert not figure.data
    assert figure.layout.annotations[0].text == "No missing values to plot."
