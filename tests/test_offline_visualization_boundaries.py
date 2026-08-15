"""Regression tests for offline rendering and the DQT dependency boundary."""

from __future__ import annotations

import builtins
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import plot_mice_convergence
from missingly.visualisation import _base, static


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    """Close matplotlib figures created by each rendering regression test.

    Returns
    -------
    None
        The fixture closes figures after the test body completes.

    Examples
    --------
    This fixture is applied automatically to this module's tests.
    """
    yield
    plt.close("all")


def test_rtl_plotting_never_uses_network_or_creates_a_user_font_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Render RTL labels using only local fonts when no suitable font exists.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to deny network access and force local-font fallback.
    tmp_path : pathlib.Path
        Isolated location that would expose any attempted cache write.

    Returns
    -------
    None
        The assertions verify the rendering boundary.

    Examples
    --------
    The test exercises ``static.bar`` with a Persian column name.
    """
    cache_path = tmp_path / "user-cache"
    monkeypatch.setattr(_base, "_RTL_FONT_REGISTERED", None)
    monkeypatch.setattr(_base, "_RTL_WARNED", False)
    monkeypatch.setattr(_base, "_find_rtl_font", lambda: None)
    monkeypatch.setattr(_base, "_BUNDLED_VAZIRMATN_PATH", tmp_path / "missing.ttf", raising=False)
    monkeypatch.setattr(_base, "_FONT_CACHE_DIR", cache_path, raising=False)

    def _deny_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Rendering must not request a font over the network.")

    monkeypatch.setattr(urllib.request, "urlopen", _deny_network)

    with pytest.warns(UserWarning, match="No RTL-capable font"):
        axis = static.bar(pd.DataFrame({"سن": [1.0, np.nan]}))

    assert axis is not None
    assert not cache_path.exists()


def test_static_heatmap_never_imports_dqt_while_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the core static renderer independent from its sister package.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to reject an attempted DQT import.

    Returns
    -------
    None
        The renderer returns normally without importing DQT.

    Examples
    --------
    The test passes a two-column frame with distinct missingness patterns.
    """
    original_import = builtins.__import__

    def _guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("data_quality_toolkit"):
            raise AssertionError("Core static plotting must not import DQT.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    axis = static.heatmap(
        pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    )

    assert axis is not None


def test_legacy_shims_do_not_import_dqt_when_missingly_is_imported() -> None:
    """Allow deprecated APIs to remain lazy during the SemVer transition.

    Returns
    -------
    None
        The package import leaves DQT absent from ``sys.modules``.

    Examples
    --------
    This test removes cached DQT modules before importing the legacy shims.
    """
    saved_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("data_quality_toolkit")
    }
    try:
        __import__("missingly.manipulation")
        __import__("missingly.performance")
        __import__("missingly.stats_extra")
        assert not any(name.startswith("data_quality_toolkit") for name in sys.modules)
    finally:
        for name in list(sys.modules):
            if name.startswith("data_quality_toolkit"):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


def test_empty_mice_convergence_plot_emits_warning_without_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replace operational print output with a warning that callers can handle.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Fixture that captures process-standard output.

    Returns
    -------
    None
        The assertions verify the warning and empty stdout contract.

    Examples
    --------
    An empty convergence result represents a request with no variables to plot.
    """
    with pytest.warns(UserWarning, match="No variables to plot"):
        plot_mice_convergence({"trace_data": {}, "rhat": {}})

    assert capsys.readouterr().out == ""
