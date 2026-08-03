"""Integration tests: missingly.visualise.heatmap <-> DQT contract.

These tests verify that:
* `data_quality_toolkit.visualization.correlation_heatmap` is importable
  and has the stable signature missingly depends on.
* `missingly.visualise.heatmap` (static mode) delegates the seaborn draw
  call through logic that is consistent with DQT's implementation.
* The lazy-import pattern is intact: DQT is NOT imported at module level
  in missingly.visualise.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import data_quality_toolkit
    HAS_DQT = True
except ImportError:
    HAS_DQT = False

pytestmark = pytest.mark.skipif(
    not HAS_DQT,
    reason="data_quality_toolkit not installed"
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((30, 4)), columns=["a", "b", "c", "d"])
    df.loc[0:5, "a"] = np.nan
    df.loc[3:8, "b"] = np.nan
    df.loc[10:12, "d"] = np.nan
    return df


# ---------------------------------------------------------------------------
# DQT contract: importability and signature
# ---------------------------------------------------------------------------

def test_dqt_correlation_heatmap_importable():
    from data_quality_toolkit.visualization import correlation_heatmap  # noqa: F401


def test_dqt_correlation_heatmap_signature():
    """Stable parameters that missingly relies on must be present."""
    from data_quality_toolkit.visualization import correlation_heatmap
    sig = inspect.signature(correlation_heatmap)
    params = set(sig.parameters.keys())
    required = {"corr", "ax", "mask_insignificant", "significance",
                "cmap", "annot", "fmt", "vmin", "vmax", "n_obs"}
    missing = required - params
    assert not missing, f"DQT.correlation_heatmap is missing parameters: {missing}"


def test_dqt_correlation_heatmap_in_all():
    from data_quality_toolkit import visualization
    assert "correlation_heatmap" in visualization.__all__


# ---------------------------------------------------------------------------
# Lazy import: DQT must NOT be imported at missingly.visualise module level
# ---------------------------------------------------------------------------

def test_visualise_module_does_not_import_dqt_at_module_level():
    """Importing missingly.visualise must not drag in data_quality_toolkit
    as a side-effect at module level."""
    # Remove only DQT's cached modules. Reloading the facade in place avoids
    # replacing the root ``missingly`` module and invalidating public aliases
    # held by tests or user code elsewhere in the process.
    mods_to_remove = [k for k in sys.modules if "data_quality_toolkit" in k]
    saved = {k: sys.modules.pop(k) for k in mods_to_remove}

    try:
        visualise_module = importlib.import_module("missingly.visualise")
        importlib.reload(visualise_module)

        dqt_imported = any(
            k.startswith("data_quality_toolkit") for k in sys.modules
        )
        assert not dqt_imported, (
            "data_quality_toolkit was imported at module level in "
            "missingly.visualise — it must be a lazy function-level import."
        )
    finally:
        for module_name in list(sys.modules):
            if module_name.startswith("data_quality_toolkit"):
                sys.modules.pop(module_name)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# missingly.visualise.heatmap static mode: end-to-end
# ---------------------------------------------------------------------------

def test_heatmap_static_returns_axes(df_with_missing):
    import matplotlib.axes
    from missingly import visualise
    ax = visualise.heatmap(df_with_missing)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_heatmap_static_no_warnings(df_with_missing):
    from missingly import visualise
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        visualise.heatmap(df_with_missing)


def test_heatmap_mask_insignificant(df_with_missing):
    """mask_insignificant=True path in static mode must not raise."""
    import matplotlib.axes
    from missingly import visualise
    ax = visualise.heatmap(df_with_missing, mask_insignificant=True, significance=0.05)
    assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# DQT.correlation_heatmap direct call (mirrors what heatmap() delegates to)
# ---------------------------------------------------------------------------

def test_dqt_heatmap_on_nullity_corr(df_with_missing):
    """Call DQT directly with the nullity corr matrix that missingly builds."""
    import matplotlib.axes
    from data_quality_toolkit.visualization import correlation_heatmap

    null_mat = df_with_missing.isnull().astype(float)
    corr = null_mat.corr()
    ax = correlation_heatmap(corr)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_dqt_heatmap_mask_insignificant(df_with_missing):
    from data_quality_toolkit.visualization import correlation_heatmap

    null_mat = df_with_missing.isnull().astype(float)
    corr = null_mat.corr()
    ax = correlation_heatmap(corr, mask_insignificant=True, n_obs=len(df_with_missing))
    assert ax is not None
