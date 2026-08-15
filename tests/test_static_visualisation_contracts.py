"""Behavioral contracts for static missing-data visualisations.

The tests assert artist state and public return contracts instead of rendered
pixels, so they remain robust across supported matplotlib versions.
"""

from __future__ import annotations

import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

matplotlib.use("Agg")

from missingly.visualisation import static


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close matplotlib figures created by each independent contract test."""
    yield
    plt.close("all")


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Return mixed-dtype data with controlled nullity correlations."""
    return pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan],
            "category": pd.Categorical(
                ["a", pd.NA, "b", pd.NA, "a", pd.NA, "b", pd.NA],
                categories=["a", "b"],
                ordered=True,
            ),
            "independent": [np.nan, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, np.nan],
        }
    )


def _quadmesh_mask(ax: Axes) -> np.ndarray:
    """Return the seaborn heatmap mask from its QuadMesh artist state."""
    array = ax.collections[0].get_array()
    return np.ma.getmaskarray(array).reshape(3, 3)


def test_static_plots_reuse_provided_axes_without_network_or_input_mutation(
    monkeypatch,
    mixed_frame,
) -> None:
    """Core static plots must be local, deterministic, and non-mutating."""
    original = mixed_frame.copy(deep=True)

    def reject_connections(*args, **kwargs):
        """Fail if a static plot tries to open a network connection."""
        raise AssertionError("static visualisation must not use the network")

    monkeypatch.setattr(socket.socket, "connect", reject_connections)
    fig, axes = plt.subplots(1, 2)

    assert static.matrix(mixed_frame, ax=axes[0]) is axes[0]
    assert static.bar(mixed_frame, ax=axes[1]) is axes[1]
    pd.testing.assert_frame_equal(mixed_frame, original)


def test_heatmap_honours_significance_mask_and_preserves_constant_nullity(
    mixed_frame,
) -> None:
    """Significance masking hides only unsupported correlations, not data errors."""
    masked = static.heatmap(
        mixed_frame,
        annot=False,
        mask_insignificant=True,
        significance=0.05,
    )
    unmasked = static.heatmap(
        mixed_frame,
        annot=False,
        mask_insignificant=False,
    )
    constant = pd.DataFrame(
        {
            "always_missing": [np.nan] * 8,
            "alternating": [np.nan, 1.0] * 4,
            "also_always_missing": [np.nan] * 8,
        }
    )
    constant_ax = static.heatmap(constant, annot=False, mask_insignificant=True)

    masked_cells = _quadmesh_mask(masked)
    unmasked_cells = _quadmesh_mask(unmasked)

    assert masked_cells[0, 2]
    assert not unmasked_cells[0, 2]
    assert np.all(np.diag(masked_cells))
    assert np.all(_quadmesh_mask(constant_ax))


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"complete": [1.0, 2.0]}),
        pd.DataFrame({"complete": [1.0, 2.0], "partial": [np.nan, 2.0]}),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "kendall"}, "method must be either"),
        (
            {"mask_insignificant": True, "significance": 0.0},
            "significance must be in the interval",
        ),
    ],
)
def test_heatmap_validates_public_options_before_shape_shortcuts(
    frame,
    kwargs,
    message,
) -> None:
    """Invalid heatmap options must fail before data-dependent early returns."""
    with pytest.raises(ValueError, match=message):
        static.heatmap(frame, **kwargs)


def test_heatmap_forwards_validated_semantics_to_plotly_backend(monkeypatch) -> None:
    """Interactive heatmap output must preserve the static option contract."""
    captured: dict[str, object] = {}
    sentinel = object()

    def return_plotly_sentinel(frame, **kwargs):
        """Capture forwarded public options without importing Plotly."""
        captured["frame"] = frame
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(static, "_require_plotly", lambda: None)
    monkeypatch.setattr(
        "missingly.visualisation.interactive._heatmap_plotly",
        return_plotly_sentinel,
    )
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})

    result = static.heatmap(
        frame,
        backend="plotly",
        method="phi",
        mask_insignificant=True,
        significance=0.1,
        custom_option="preserve-me",
    )

    assert result is sentinel
    assert captured["method"] == "phi"
    assert captured["mask_insignificant"] is True
    assert captured["significance"] == 0.1
    assert captured["custom_option"] == "preserve-me"


@pytest.mark.parametrize("plotter", [static.heatmap, static.dendrogram])
def test_insufficient_nullity_plots_reuse_caller_axes(plotter) -> None:
    """Empty/nullity-insufficient visualisations must honour the supplied axes."""
    frame = pd.DataFrame({"complete": [1.0, 2.0], "other": [3.0, 4.0]})
    fig, ax = plt.subplots()

    result = plotter(frame, ax=ax)

    assert result is ax
    assert "Not enough columns" in ax.texts[0].get_text()


def test_parallel_coordinates_reuses_caller_axes_for_non_numeric_data() -> None:
    """The non-numeric empty-state must not create an unrelated figure."""
    frame = pd.DataFrame({"label": ["a", "b"], "group": ["x", "y"]})
    fig, ax = plt.subplots()

    result = static.vis_parallel_coords(frame, ax=ax)

    assert result is ax
    assert "No numeric columns" in ax.texts[0].get_text()


@pytest.mark.parametrize(
    ("frame", "min_subset_size"),
    [
        (pd.DataFrame({"complete": [1.0, 2.0]}), 1),
        (pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 1.0]}), 3),
    ],
)
def test_upset_empty_states_keep_the_documented_axes_schema(
    frame,
    min_subset_size,
) -> None:
    """UpSet must return its three axes even when no pattern is drawable."""
    result = static.upset(frame, min_subset_size=min_subset_size)

    assert set(result) == {"intersections", "matrix", "totals"}
    assert all(isinstance(axis, Axes) for axis in result.values())
    assert "No missingness patterns" in result["intersections"].texts[0].get_text()


def test_upset_rejects_unknown_static_options_before_empty_shortcut() -> None:
    """UpSet option validation must not depend on whether patterns exist."""
    with pytest.raises(TypeError, match="Unsupported UpSet keyword arguments: unknown"):
        static.upset(pd.DataFrame({"complete": [1.0, 2.0]}), unknown=True)


@pytest.mark.parametrize("function_name", ["vis_miss", "miss_cluster"])
def test_clustering_fallback_warns_for_recoverable_value_error(
    monkeypatch,
    mixed_frame,
    function_name,
) -> None:
    """Degenerate cluster input keeps source order and exposes the fallback."""
    def fail_linkage(*args, **kwargs):
        """Reproduce scipy's recoverable invalid-distance error."""
        raise ValueError("synthetic invalid linkage input")

    if function_name == "vis_miss":
        monkeypatch.setattr("scipy.cluster.hierarchy.linkage", fail_linkage)
        with pytest.warns(UserWarning, match="original order"):
            result = static.vis_miss(mixed_frame, cluster=True)
    else:
        monkeypatch.setattr("scipy.cluster.hierarchy.linkage", fail_linkage)
        with pytest.warns(UserWarning, match="original order"):
            result = static.miss_cluster(mixed_frame)

    assert isinstance(result, Axes)


@pytest.mark.parametrize("function_name", ["vis_miss", "miss_cluster"])
def test_clustering_does_not_hide_unexpected_backend_errors(
    monkeypatch,
    mixed_frame,
    function_name,
) -> None:
    """Unexpected clustering errors are programming errors, not fallback cases."""
    def fail_linkage(*args, **kwargs):
        """Reproduce an unsupported backend failure."""
        raise RuntimeError("unexpected linkage backend failure")

    monkeypatch.setattr("scipy.cluster.hierarchy.linkage", fail_linkage)

    with pytest.raises(RuntimeError, match="unexpected linkage backend failure"):
        if function_name == "vis_miss":
            static.vis_miss(mixed_frame, cluster=True)
        else:
            static.miss_cluster(mixed_frame)
