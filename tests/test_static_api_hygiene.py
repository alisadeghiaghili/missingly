"""API-hygiene contracts for the static visualisation functions.

Unsupported keyword arguments, ignored styling options, misleading labels and
the interactive-extra install hint must all be reported truthfully while the
public call signatures keep working.
"""

from __future__ import annotations

import builtins
import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from missingly.visualisation import static


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close matplotlib figures created by each test."""
    yield
    plt.close("all")


@pytest.fixture
def frame() -> pd.DataFrame:
    """Small frame with missing values in three columns."""
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
            "b": [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0],
            "c": [1.0, 2.0, np.nan, 4.0, np.nan, np.nan],
            "g": ["x", "x", "y", "y", "x", "y"],
        }
    )


# Functions whose matplotlib path discards unknown keyword arguments.
SWALLOWING = {
    "matrix": lambda f, **kw: static.matrix(f, **kw),
    "bar": lambda f, **kw: static.bar(f, **kw),
    "miss_case": lambda f, **kw: static.miss_case(f, **kw),
    "miss_var_pct": lambda f, **kw: static.miss_var_pct(f, **kw),
    "miss_which": lambda f, **kw: static.miss_which(f, **kw),
    "miss_patterns": lambda f, **kw: static.miss_patterns(f, **kw),
    "vis_miss": lambda f, **kw: static.vis_miss(f, **kw),
    "miss_cluster": lambda f, **kw: static.miss_cluster(f, **kw),
    "miss_row_profile": lambda f, **kw: static.miss_row_profile(f, **kw),
    "shadow_scatter": lambda f, **kw: static.shadow_scatter(f, "a", "b", **kw),
    "scatter_miss": lambda f, **kw: static.scatter_miss(f, "a", "b", **kw),
    "vis_miss_fct": lambda f, **kw: static.vis_miss_fct(f, "g", **kw),
    "vis_parallel_coords": lambda f, **kw: static.vis_parallel_coords(f, **kw),
}

# The function whose warning is expected for aliases and wrappers.
WARNING_SOURCE = {"vis_miss_fct": "vis_miss", "scatter_miss": "shadow_scatter"}

# Functions that forward ``**kwargs`` to a documented third-party target.
FORWARDING = {
    "heatmap",
    "miss_cooccurrence",
    "dendrogram",
    "vis_miss_by_group",
    "vis_impute_dist",
    "miss_impute_compare",
    "vis_miss_cumsum_var",
    "vis_miss_cumsum_case",
    "vis_miss_span",
}

# upset already raises TypeError for unsupported static keywords.
RAISING = {"upset"}


def _public_functions() -> set[str]:
    """Return the names of the public functions defined in ``static``."""
    return {
        name
        for name, obj in vars(static).items()
        if inspect.isfunction(obj)
        and obj.__module__ == static.__name__
        and not name.startswith("_")
    }


def test_every_public_static_function_is_classified() -> None:
    """A new public function must be added to one of the kwargs policies."""
    classified = set(SWALLOWING) | FORWARDING | RAISING
    assert _public_functions() == classified


@pytest.mark.parametrize("name", sorted(SWALLOWING))
def test_unknown_keyword_warns_and_keeps_working(name, frame) -> None:
    """Unknown keywords warn with FutureWarning naming the argument."""
    with pytest.warns(FutureWarning, match=r"bogus_option.*TypeError"):
        result = SWALLOWING[name](frame, bogus_option=1)

    assert result is not None


@pytest.mark.parametrize("name", sorted(SWALLOWING))
def test_unknown_keyword_warning_names_the_function(name, frame) -> None:
    """The warning identifies the public function that ignored the argument."""
    expected = WARNING_SOURCE.get(name, name)
    with pytest.warns(FutureWarning, match=rf"{expected}\(\)"):
        SWALLOWING[name](frame, bogus_option=1)


def test_known_keywords_do_not_warn(frame, recwarn) -> None:
    """Supported arguments never trigger the unsupported-keyword warning."""
    static.bar(frame, color="red", fontsize=9)

    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


@pytest.mark.parametrize(
    "name", ["miss_which", "miss_cluster", "miss_row_profile", "shadow_scatter"]
)
def test_interactive_without_plotly_path_warns(name, frame) -> None:
    """interactive=True on functions without a Plotly path is reported."""
    with pytest.warns(FutureWarning, match=r"interactive"):
        result = SWALLOWING[name](frame, interactive=True)

    assert isinstance(result, Axes)


def test_unknown_backend_value_warns_and_uses_matplotlib(frame) -> None:
    """A misspelled backend is reported instead of silently falling back."""
    with pytest.warns(FutureWarning, match=r"backend.*'Plotly'"):
        result = static.bar(frame, backend="Plotly")

    assert isinstance(result, Axes)


@pytest.mark.parametrize(
    ("plotter", "kwargs", "dropped"),
    [
        (static.bar, {"color": "red", "log": True}, "color"),
        (static.miss_case, {"color": "red", "figsize": (3, 3)}, "color"),
        (static.matrix, {"fontsize": 20}, "fontsize"),
        (static.heatmap, {"cmap": "viridis", "annot": False}, "cmap"),
    ],
)
def test_styling_dropped_on_plotly_path_is_reported(
    plotter, kwargs, dropped, frame
) -> None:
    """Styling options the Plotly backend cannot honour are named in a warning."""
    pytest.importorskip("plotly")
    with pytest.warns(UserWarning, match=rf"ignored.*{dropped}"):
        figure = plotter(frame, interactive=True, **kwargs)

    assert hasattr(figure, "data")


def test_default_styling_on_plotly_path_does_not_warn(frame, recwarn) -> None:
    """Untouched defaults must not produce an ignored-argument warning."""
    pytest.importorskip("plotly")
    static.bar(frame, interactive=True)

    assert not [w for w in recwarn if "ignored" in str(w.message)]


def _tick_sizes(ax: Axes) -> set[float]:
    """Return the font sizes of all tick labels on both axes."""
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    return {label.get_fontsize() for label in labels}


@pytest.mark.parametrize(
    "plotter",
    [
        lambda f: static.heatmap(f, fontsize=17, mask_insignificant=False),
        lambda f: static.miss_cooccurrence(f, fontsize=17),
        lambda f: static.vis_miss_by_group(f, "g", fontsize=17),
    ],
)
def test_fontsize_is_applied_to_tick_labels(plotter, frame) -> None:
    """The documented fontsize controls the tick-label size."""
    ax = plotter(frame)

    assert _tick_sizes(ax) == {17.0}


@pytest.mark.parametrize(
    ("orientation", "axis"),
    [("left", "y"), ("right", "y"), ("top", "x"), ("bottom", "x")],
)
def test_dendrogram_fontsize_targets_the_label_axis(orientation, axis, frame) -> None:
    """Leaf labels sit on x for top/bottom and on y for left/right."""
    ax = static.dendrogram(frame, fontsize=15, orientation=orientation)

    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    texts = [label for label in labels if label.get_text()]
    assert texts
    assert {label.get_fontsize() for label in texts} == {15.0}


@pytest.mark.parametrize("colour", ["red", "#ff0000", (1.0, 0.0, 0.0)])
def test_matrix_accepts_matplotlib_colours(colour, frame) -> None:
    """Colour strings are accepted like in every other static function."""
    ax = static.matrix(frame, color=colour, sparkline=False)

    rgba = ax.images[0].get_array()
    assert tuple(rgba[0, 0][:3]) == (1.0, 0.0, 0.0)
    assert tuple(rgba[1, 0][:3]) == (1.0, 1.0, 1.0)


def test_matrix_sparkline_uses_the_converted_colour(frame) -> None:
    """The sparkline line shares the converted RGB colour."""
    static.matrix(frame, color="red")
    spark = plt.gcf().axes[1]

    assert mcolors.to_rgb(spark.lines[0].get_color()) == (1.0, 0.0, 0.0)


def test_matrix_rejects_an_invalid_colour_clearly(frame) -> None:
    """An invalid colour fails with matplotlib's explicit message."""
    with pytest.raises(ValueError, match="not-a-colour"):
        static.matrix(frame, color="not-a-colour")


def test_miss_case_x_label_describes_sorted_ranks(frame) -> None:
    """Sorted bars are ranks, not row indices."""
    sorted_ax = static.miss_case(frame, sort=True)
    unsorted_ax = static.miss_case(frame, sort=False)

    assert "sorted" in sorted_ax.get_xlabel().lower()
    assert unsorted_ax.get_xlabel() == "Row index"


def test_parallel_coords_title_does_not_name_a_colour(frame) -> None:
    """The legend carries the colours, so the title must not hard-code one."""
    ax = static.vis_parallel_coords(frame.drop(columns="g"), color_missing="green")

    assert "orange" not in ax.get_title().lower()


def _block_plotly(monkeypatch) -> None:
    """Make every plotly import fail."""
    original_import = builtins.__import__

    def import_without_plotly(name, *args, **kwargs):
        if name.split(".")[0] == "plotly":
            raise ImportError("Plotly intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plotly)


@pytest.mark.parametrize("name", ["bar", "heatmap", "miss_case", "vis_miss"])
def test_public_interactive_call_names_the_extra(name, frame, monkeypatch) -> None:
    """The ImportError users see names the missingly[interactive] extra."""
    _block_plotly(monkeypatch)

    with pytest.raises(ImportError, match=r"missingly\[interactive\]"):
        getattr(static, name)(frame, interactive=True)
