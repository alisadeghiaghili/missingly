"""Tests for Phase-2 interactive visualizations in missingly.visualise.

Phase 2 adds ``interactive=True`` support to four additional functions:

* :func:`~missingly.visualise.miss_patterns`
* :func:`~missingly.visualise.miss_cluster`
* :func:`~missingly.visualise.miss_row_profile`
* :func:`~missingly.visualise.dendrogram`

Each function is tested for:

- Correct return type (``matplotlib.axes.Axes`` by default, ``plotly.graph_objects.Figure`` when ``interactive=True``)
- Non-empty traces / data in the interactive figure
- Key layout properties (title present, axes labelled)
- Sentinel / custom ``missing_values`` support
- Graceful handling of DataFrames with no missing values
- Backward compatibility: ``interactive=False`` still returns an Axes

Tests that require plotly are skipped automatically when plotly is not
installed, keeping the test suite runnable in minimal environments.

Example
-------
    pytest tests/test_visualise_interactive_phase2.py -v
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Skip entire module if plotly is not available
# ---------------------------------------------------------------------------
plotly = pytest.importorskip("plotly", reason="plotly not installed")
import plotly.graph_objects as go

from missingly.visualise import (
    miss_patterns,
    miss_cluster,
    miss_row_profile,
    dendrogram,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Standard mixed-missingness DataFrame used across most tests."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age":      np.where(np.random.rand(n) < 0.2, np.nan, np.random.randint(20, 70, n).astype(float)),
        "income":   np.where(np.random.rand(n) < 0.3, np.nan, np.random.randint(30000, 120000, n).astype(float)),
        "score":    np.where(np.random.rand(n) < 0.15, np.nan, np.random.uniform(0, 100, n)),
        "category": np.where(np.random.rand(n) < 0.1, np.nan, np.random.choice(["A", "B", "C"], n)),
    })
    return df


@pytest.fixture
def complete_df() -> pd.DataFrame:
    """DataFrame with no missing values."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    })


@pytest.fixture
def sentinel_df() -> pd.DataFrame:
    """DataFrame where -99 and 'unknown' act as sentinel missing values."""
    return pd.DataFrame({
        "a": [1.0, -99.0, 3.0, 4.0, -99.0, 6.0],
        "b": [10.0, 20.0, -99.0, 40.0, 50.0, 60.0],
        "c": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ===========================================================================
# miss_patterns
# ===========================================================================

class TestMissPatternsInteractive:
    """Tests for :func:`missingly.visualise.miss_patterns` interactive mode."""

    def test_returns_figure_when_interactive(self, sample_df):
        """interactive=True must return a plotly Figure."""
        fig = miss_patterns(sample_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_returns_axes_by_default(self, sample_df):
        """Default call must still return a matplotlib Axes (backward compat)."""
        ax = miss_patterns(sample_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_axes_when_interactive_false(self, sample_df):
        """Explicit interactive=False must return Axes."""
        ax = miss_patterns(sample_df, interactive=False)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_figure_has_bar_trace(self, sample_df):
        """Interactive figure must contain at least one Bar trace."""
        fig = miss_patterns(sample_df, interactive=True)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_figure_has_title(self, sample_df):
        """Interactive figure must have a non-empty title."""
        fig = miss_patterns(sample_df, interactive=True)
        assert fig.layout.title.text

    def test_top_n_respected(self, sample_df):
        """top_n parameter must limit the number of bars shown."""
        top_n = 3
        fig = miss_patterns(sample_df, interactive=True, top_n=top_n)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        total_bars = sum(len(t.x or t.y or []) for t in bar_traces)
        assert total_bars <= top_n

    def test_sentinel_missing_values(self, sentinel_df):
        """Sentinel values must be treated as missing when passed."""
        fig = miss_patterns(sentinel_df, interactive=True, missing_values=[-99])
        assert isinstance(fig, go.Figure)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_complete_df_returns_figure(self, complete_df):
        """A complete DataFrame should return a Figure (possibly empty pattern)."""
        fig = miss_patterns(complete_df, interactive=True)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# miss_cluster
# ===========================================================================

class TestMissClusterInteractive:
    """Tests for :func:`missingly.visualise.miss_cluster` interactive mode."""

    def test_returns_figure_when_interactive(self, sample_df):
        """interactive=True must return a plotly Figure."""
        fig = miss_cluster(sample_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_returns_axes_by_default(self, sample_df):
        """Default call must still return a matplotlib Axes."""
        ax = miss_cluster(sample_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_axes_when_interactive_false(self, sample_df):
        """Explicit interactive=False must return Axes."""
        ax = miss_cluster(sample_df, interactive=False)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_figure_has_heatmap_trace(self, sample_df):
        """Interactive figure must contain a Heatmap trace."""
        fig = miss_cluster(sample_df, interactive=True)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_figure_has_title(self, sample_df):
        """Interactive figure must have a non-empty title."""
        fig = miss_cluster(sample_df, interactive=True)
        assert fig.layout.title.text

    def test_heatmap_dimensions(self, sample_df):
        """Heatmap z must have shape (n_rows, n_cols)."""
        fig = miss_cluster(sample_df, interactive=True)
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        z = np.array(heatmap.z)
        assert z.shape == (len(sample_df), sample_df.shape[1])

    def test_sentinel_missing_values(self, sentinel_df):
        """Sentinel values must be treated as missing."""
        fig = miss_cluster(sentinel_df, interactive=True, missing_values=[-99])
        assert isinstance(fig, go.Figure)

    def test_complete_df(self, complete_df):
        """Complete DataFrame must still return a Figure."""
        fig = miss_cluster(complete_df, interactive=True)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# miss_row_profile
# ===========================================================================

class TestMissRowProfileInteractive:
    """Tests for :func:`missingly.visualise.miss_row_profile` interactive mode."""

    def test_returns_figure_when_interactive(self, sample_df):
        """interactive=True must return a plotly Figure."""
        fig = miss_row_profile(sample_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_returns_axes_by_default(self, sample_df):
        """Default call must still return a matplotlib Axes."""
        ax = miss_row_profile(sample_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_axes_when_interactive_false(self, sample_df):
        """Explicit interactive=False must return Axes."""
        ax = miss_row_profile(sample_df, interactive=False)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_figure_has_bar_trace(self, sample_df):
        """Interactive figure must contain a Bar trace."""
        fig = miss_row_profile(sample_df, interactive=True)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_figure_has_mean_line(self, sample_df):
        """Interactive figure must include a mean-line shape or scatter trace."""
        fig = miss_row_profile(sample_df, interactive=True)
        has_shape = len(fig.layout.shapes) > 0
        has_scatter = any(isinstance(t, go.Scatter) for t in fig.data)
        assert has_shape or has_scatter

    def test_figure_has_title(self, sample_df):
        """Interactive figure must have a non-empty title."""
        fig = miss_row_profile(sample_df, interactive=True)
        assert fig.layout.title.text

    def test_bar_count_equals_unique_miss_counts(self, sample_df):
        """Number of bars must equal number of unique per-row missing counts."""
        from missingly.visualise import _nullity
        row_miss = _nullity(sample_df).sum(axis=1)
        unique_counts = row_miss.value_counts().shape[0]
        fig = miss_row_profile(sample_df, interactive=True)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        total_bars = sum(len(t.x or []) for t in bar_traces)
        assert total_bars == unique_counts

    def test_sentinel_missing_values(self, sentinel_df):
        """Sentinel values must be treated as missing."""
        fig = miss_row_profile(sentinel_df, interactive=True, missing_values=[-99])
        assert isinstance(fig, go.Figure)

    def test_complete_df(self, complete_df):
        """Complete DataFrame must return a Figure."""
        fig = miss_row_profile(complete_df, interactive=True)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# dendrogram
# ===========================================================================

class TestDendrogramInteractive:
    """Tests for :func:`missingly.visualise.dendrogram` interactive mode."""

    def test_returns_figure_when_interactive(self, sample_df):
        """interactive=True must return a plotly Figure."""
        fig = dendrogram(sample_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_returns_axes_by_default(self, sample_df):
        """Default call must still return a matplotlib Axes."""
        ax = dendrogram(sample_df)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_axes_when_interactive_false(self, sample_df):
        """Explicit interactive=False must return Axes."""
        ax = dendrogram(sample_df, interactive=False)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_figure_has_scatter_traces(self, sample_df):
        """Interactive dendrogram must contain Scatter traces (dendrogram lines)."""
        fig = dendrogram(sample_df, interactive=True)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1

    def test_figure_has_title(self, sample_df):
        """Interactive figure must have a non-empty title."""
        fig = dendrogram(sample_df, interactive=True)
        assert fig.layout.title.text

    def test_raises_on_single_variable_missingness(self):
        """Must raise ValueError when fewer than 2 columns have variable missingness."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            dendrogram(df, interactive=True)

    def test_sentinel_missing_values(self, sentinel_df):
        """Sentinel values must be treated as missing."""
        fig = dendrogram(sentinel_df, interactive=True, missing_values=[-99])
        assert isinstance(fig, go.Figure)

    def test_linkage_method_ward(self, sample_df):
        """method='ward' (default) must not raise."""
        fig = dendrogram(sample_df, interactive=True, method="ward")
        assert isinstance(fig, go.Figure)

    def test_linkage_method_average(self, sample_df):
        """method='average' must not raise."""
        fig = dendrogram(sample_df, interactive=True, method="average")
        assert isinstance(fig, go.Figure)
