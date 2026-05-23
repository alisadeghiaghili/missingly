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
import types
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
    """Tiny 8×4 DataFrame with controlled missingness."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 4))
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    # Introduce known missingness
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
    # Save originals
    saved = {k: v for k, v in sys.modules.items() if k.startswith("plotly")}
    # Block plotly
    monkeypatch.setitem(sys.modules, "plotly", None)  # None => ImportError
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "plotly.subplots", None)
    yield
    # Restore (monkeypatch handles teardown automatically, but restore anyway)
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
# Phase 2 interactive functions — return type checks
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
        asse