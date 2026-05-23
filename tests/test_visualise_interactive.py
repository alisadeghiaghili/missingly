"""Tests for the interactive (Plotly) backends of visualise functions.

Coverage strategy
-----------------
* When plotly IS installed (normal CI): assert return type is
  ``plotly.graph_objects.Figure``.
* When plotly is NOT installed: assert ``ImportError`` is raised with a
  helpful installation hint (monkeypatched via ``sys.modules``).

Functions under test (Phase 1 + Phase 2 interactive functions)
--------------------------------------------------------------
Phase 1 (existing): vis_miss, heatmap, matrix, miss_var_pct, miss_cooccurrence
Phase 2 (new):      bar, miss_case, upset, miss_patterns
"""

from __future__ import annotations

import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from missingly import visualise


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df() -> pd.DataFrame:
    """Tiny 8x4 DataFrame with controlled missingness."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 4))
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    df.loc[0, "a"] = np.nan
    df.loc[1, "b"] = np.nan
    df.loc[2, "b"] = np.nan
    df.loc[3, "c"] = np.nan
    df.loc[0, "c"] = np.nan
    df.loc[4, "d"] = np.nan
    return df


@pytest.fixture
def no_plotly(monkeypatch):
    """Monkeypatch sys.modules so that 'import plotly' raises ImportError."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("plotly")}
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "plotly.subplots", None)
    yield
    for k in list(sys.modules.keys()):
        if k.startswith("plotly"):
            if k in saved:
                sys.modules[k] = saved[k]
            else:
                del sys.modules[k]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_plotly() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Phase 2 interactive functions -- return type checks (plotly present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPhase2InteractiveReturnType:
    """Assert interactive=True returns a plotly Figure when plotly is present."""

    def test_bar_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.bar(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_case_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.miss_case(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_upset_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.upset(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_patterns_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.miss_patterns(small_df, interactive=True)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Phase 2 -- static fallback still returns Axes (not a Figure)
# ---------------------------------------------------------------------------

class TestPhase2StaticFallback:
    """Ensure interactive=False (default) still returns matplotlib Axes."""

    def test_bar_static(self, small_df):
        import matplotlib.axes
        result = visualise.bar(small_df)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_miss_case_static(self, small_df):
        import matplotlib.axes
        result = visualise.miss_case(small_df)
        assert isinstance(result, matplotlib.axes.Axes)

    def test_upset_static_returns_dict(self, small_df):
        result = visualise.upset(small_df)
        assert isinstance(result, dict)
        assert "intersections" in result
        assert "matrix" in result
        assert "totals" in result

    def test_miss_patterns_static(self, small_df):
        import matplotlib.axes
        result = visualise.miss_patterns(small_df)
        assert isinstance(result, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Phase 2 -- ImportError when plotly is absent
# ---------------------------------------------------------------------------

class TestPhase2ImportErrorWithoutPlotly:
    """When plotly is not installed, interactive=True raises ImportError."""

    def test_bar_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.bar(small_df, interactive=True)

    def test_miss_case_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.miss_case(small_df, interactive=True)

    def test_upset_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.upset(small_df, interactive=True)

    def test_miss_patterns_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.miss_patterns(small_df, interactive=True)


# ---------------------------------------------------------------------------
# Phase 1 interactive functions -- regression guard (return type)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPhase1RegressionReturnType:
    """Existing Phase 1 functions still return Figure when interactive=True."""

    def test_vis_miss_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.vis_miss(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_heatmap_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.heatmap(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_matrix_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.matrix(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_var_pct_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.miss_var_pct(small_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_cooccurrence_returns_figure(self, small_df):
        import plotly.graph_objects as go
        fig = visualise.miss_cooccurrence(small_df, interactive=True)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Phase 1 -- ImportError regression guard
# ---------------------------------------------------------------------------

class TestPhase1ImportErrorWithoutPlotly:
    """Phase 1 functions also raise ImportError with helpful message."""

    def test_vis_miss_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.vis_miss(small_df, interactive=True)

    def test_heatmap_import_error(self, small_df, no_plotly):
        with pytest.raises(ImportError, match="pip install plotly"):
            visualise.heatmap(small_df, interactive=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_plotly(), reason="plotly not installed")
class TestPhase2EdgeCases:
    """Edge cases: no-missing DataFrame, single-column, large index."""

    def test_bar_no_missing(self):
        """bar with no missing data should still return a Figure."""
        import plotly.graph_objects as go
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        fig = visualise.bar(df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_case_all_missing_row(self):
        """miss_case with a completely empty row should not raise."""
        import plotly.graph_objects as go
        df = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 2.0]})
        fig = visualise.miss_case(df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_upset_no_missing_returns_figure(self):
        """upset with no missing cols returns an annotated empty Figure."""
        import plotly.graph_objects as go
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fig = visualise.upset(df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_miss_patterns_top_n_respected(self):
        """miss_patterns respects top_n parameter in interactive mode."""
        import plotly.graph_objects as go
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.standard_normal((50, 6)),
            columns=list("abcdef")
        )
        # Introduce varied patterns
        for i in range(0, 50, 3):
            df.iloc[i, rng.integers(0, 6)] = np.nan
        fig = visualise.miss_patterns(df, top_n=3, interactive=True)
        assert isinstance(fig, go.Figure)
        # The y-axis should have at most 3 entries
        n_bars = len(fig.data[0].y)
        assert n_bars <= 3
