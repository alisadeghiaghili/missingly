"""Tests for interactive (Plotly) mode of Phase-1 visualisation functions.

Conventions
-----------
* Each test verifies that ``interactive=True`` returns a
  ``plotly.graph_objects.Figure`` with the correct title and at least
  one trace.
* Static (``interactive=False``) behaviour is NOT re-tested here — it
  lives in ``test_visualise.py`` and must remain 100% unchanged.
* Tests are skipped automatically when plotly is not installed, so the
  CI matrix without plotly stays green.
* Fixtures are intentionally small (4-10 rows) to keep tests fast.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

pytest.importorskip("plotly", reason="plotly not installed")

from missingly import visualise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nan_df():
    """Small DataFrame with clear NaN pattern for interactive plot tests."""
    return pd.DataFrame({
        'X': [1.0, np.nan, 3.0, np.nan, 5.0],
        'Y': [np.nan, 2.0, np.nan, 4.0, 5.0],
        'Z': [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def sentinel_df():
    """DataFrame with sentinel values instead of NaN."""
    return pd.DataFrame({
        'A': [1, 2, -99, 4],
        'B': [-99, 20.0, 30.0, 40.0],
        'C': [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def no_missing_df():
    """Complete DataFrame — no missing values anywhere."""
    return pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})


@pytest.fixture
def persian_df():
    """DataFrame with Persian column names."""
    return pd.DataFrame({
        'درآمد': [1000.0, np.nan, 3000.0, np.nan],
        'سن':   [25.0,  30.0,  np.nan,  40.0],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_figure(obj) -> bool:
    """Return True if *obj* is a plotly Figure."""
    return isinstance(obj, go.Figure)


# ---------------------------------------------------------------------------
# vis_miss — interactive
# ---------------------------------------------------------------------------

def test_vis_miss_interactive_returns_figure(nan_df):
    fig = visualise.vis_miss(nan_df, interactive=True)
    assert _is_figure(fig)


def test_vis_miss_interactive_title(nan_df):
    fig = visualise.vis_miss(nan_df, interactive=True)
    assert fig.layout.title.text == "Missing Data Overview"


def test_vis_miss_interactive_has_trace(nan_df):
    fig = visualise.vis_miss(nan_df, interactive=True)
    assert len(fig.data) >= 1


def test_vis_miss_interactive_sentinel(sentinel_df):
    fig = visualise.vis_miss(sentinel_df, missing_values=[-99], interactive=True)
    assert _is_figure(fig)


def test_vis_miss_interactive_no_missing(no_missing_df):
    fig = visualise.vis_miss(no_missing_df, interactive=True)
    assert _is_figure(fig)


def test_vis_miss_interactive_show_pct_false(nan_df):
    fig = visualise.vis_miss(nan_df, interactive=True, show_pct=False)
    assert _is_figure(fig)


def test_vis_miss_interactive_cluster(nan_df):
    fig = visualise.vis_miss(nan_df, interactive=True, cluster=True)
    assert _is_figure(fig)


def test_vis_miss_interactive_persian(persian_df):
    fig = visualise.vis_miss(persian_df, interactive=True)
    assert _is_figure(fig)


# ---------------------------------------------------------------------------
# heatmap — interactive
# ---------------------------------------------------------------------------

def test_heatmap_interactive_returns_figure(nan_df):
    fig = visualise.heatmap(nan_df, interactive=True)
    assert _is_figure(fig)


def test_heatmap_interactive_title(nan_df):
    fig = visualise.heatmap(nan_df, interactive=True)
    assert "Nullity Correlation Heatmap" in fig.layout.title.text


def test_heatmap_interactive_has_trace(nan_df):
    fig = visualise.heatmap(nan_df, interactive=True)
    assert len(fig.data) >= 1


def test_heatmap_interactive_phi(nan_df):
    fig = visualise.heatmap(nan_df, interactive=True, method="phi")
    assert "Phi" in fig.layout.title.text


def test_heatmap_interactive_pearson(nan_df):
    fig = visualise.heatmap(nan_df, interactive=True, method="pearson")
    assert "Pearson" in fig.layout.title.text


def test_heatmap_interactive_sentinel(sentinel_df):
    fig = visualise.heatmap(sentinel_df, missing_values=[-99], interactive=True)
    assert _is_figure(fig)


def test_heatmap_interactive_no_missing(no_missing_df):
    fig = visualise.heatmap(no_missing_df, interactive=True)
    assert _is_figure(fig)


def test_heatmap_interactive_persian(persian_df):
    fig = visualise.heatmap(persian_df, interactive=True)
    assert _is_figure(fig)


def test_heatmap_interactive_mask_insignificant(nan_df):
    df = pd.DataFrame({
        'A': [np.nan if i % 2 == 0 else float(i) for i in range(20)],
        'B': [np.nan if i % 2 == 0 else float(i) for i in range(20)],
        'C': [np.nan if i % 3 == 0 else float(i) for i in range(20)],
    })
    fig = visualise.heatmap(df, interactive=True, mask_insignificant=True)
    assert _is_figure(fig)


# ---------------------------------------------------------------------------
# matrix — interactive
# ---------------------------------------------------------------------------

def test_matrix_interactive_returns_figure(nan_df):
    fig = visualise.matrix(nan_df, interactive=True)
    assert _is_figure(fig)


def test_matrix_interactive_title(nan_df):
    fig = visualise.matrix(nan_df, interactive=True)
    assert fig.layout.title.text == "Missing Data Matrix"


def test_matrix_interactive_has_trace(nan_df):
    fig = visualise.matrix(nan_df, interactive=True)
    assert len(fig.data) >= 1


def test_matrix_interactive_sentinel(sentinel_df):
    fig = visualise.matrix(sentinel_df, missing_values=[-99], interactive=True)
    assert _is_figure(fig)


def test_matrix_interactive_no_missing(no_missing_df):
    fig = visualise.matrix(no_missing_df, interactive=True)
    assert _is_figure(fig)


def test_matrix_interactive_persian(persian_df):
    fig = visualise.matrix(persian_df, interactive=True)
    assert _is_figure(fig)


# ---------------------------------------------------------------------------
# miss_var_pct — interactive
# ---------------------------------------------------------------------------

def test_miss_var_pct_interactive_returns_figure(nan_df):
    fig = visualise.miss_var_pct(nan_df, interactive=True)
    assert _is_figure(fig)


def test_miss_var_pct_interactive_title(nan_df):
    fig = visualise.miss_var_pct(nan_df, interactive=True)
    assert fig.layout.title.text == "Missing Values per Variable (%)"


def test_miss_var_pct_interactive_has_trace(nan_df):
    fig = visualise.miss_var_pct(nan_df, interactive=True)
    assert len(fig.data) >= 1


def test_miss_var_pct_interactive_sorted(nan_df):
    fig = visualise.miss_var_pct(nan_df, interactive=True, sort=True)
    assert _is_figure(fig)


def test_miss_var_pct_interactive_unsorted(nan_df):
    fig = visualise.miss_var_pct(nan_df, interactive=True, sort=False)
    assert _is_figure(fig)


def test_miss_var_pct_interactive_sentinel(sentinel_df):
    fig = visualise.miss_var_pct(sentinel_df, missing_values=[-99], interactive=True)
    assert _is_figure(fig)


def test_miss_var_pct_interactive_no_missing(no_missing_df):
    fig = visualise.miss_var_pct(no_missing_df, interactive=True)
    assert _is_figure(fig)


def test_miss_var_pct_interactive_persian(persian_df):
    fig = visualise.miss_var_pct(persian_df, interactive=True)
    assert _is_figure(fig)


# ---------------------------------------------------------------------------
# miss_cooccurrence — interactive
# ---------------------------------------------------------------------------

def test_miss_cooccurrence_interactive_returns_figure(nan_df):
    fig = visualise.miss_cooccurrence(nan_df, interactive=True)
    assert _is_figure(fig)


def test_miss_cooccurrence_interactive_title_normalized(nan_df):
    fig = visualise.miss_cooccurrence(nan_df, interactive=True, normalize=True)
    assert "fraction" in fig.layout.title.text


def test_miss_cooccurrence_interactive_title_count(nan_df):
    fig = visualise.miss_cooccurrence(nan_df, interactive=True, normalize=False)
    assert "count" in fig.layout.title.text


def test_miss_cooccurrence_interactive_has_trace(nan_df):
    fig = visualise.miss_cooccurrence(nan_df, interactive=True)
    assert len(fig.data) >= 1


def test_miss_cooccurrence_interactive_sentinel(sentinel_df):
    fig = visualise.miss_cooccurrence(sentinel_df, missing_values=[-99], interactive=True)
    assert _is_figure(fig)


def test_miss_cooccurrence_interactive_persian(persian_df):
    fig = visualise.miss_cooccurrence(persian_df, interactive=True)
    assert _is_figure(fig)


# ---------------------------------------------------------------------------
# Backward compatibility: interactive=False (default) still returns Axes
# ---------------------------------------------------------------------------

def test_vis_miss_default_still_returns_axes(nan_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    ax = visualise.vis_miss(nan_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_heatmap_default_still_returns_axes(nan_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    ax = visualise.heatmap(nan_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_matrix_default_still_returns_axes(nan_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    ax = visualise.matrix(nan_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_miss_var_pct_default_still_returns_axes(nan_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    ax = visualise.miss_var_pct(nan_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_miss_cooccurrence_default_still_returns_axes(nan_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    ax = visualise.miss_cooccurrence(nan_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


# ---------------------------------------------------------------------------
# ImportError path: _require_plotly raises helpfully
# ---------------------------------------------------------------------------

def test_require_plotly_raises_import_error(monkeypatch):
    """If plotly is missing, _require_plotly must raise ImportError with hint."""
    import sys
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"plotly": None, "plotly.graph_objects": None}):
        with pytest.raises((ImportError, TypeError)):
            visualise._require_plotly()
