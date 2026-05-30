"""Shared helpers for the visualisation layer.

This module is **private** – import from :mod:`missingly.visualisation` or
:mod:`missingly.visualise` instead.

Contents
--------
_rtl_safe
    Wrap Arabic/Persian text so matplotlib renders it correctly.
_safe_labels
    Apply :func:`_rtl_safe` to every label in a sequence.
_nullity
    Return a boolean DataFrame marking missing positions.
_pct_labels
    Column labels with missingness percentage appended.
_require_plotly
    Lazy import of ``plotly.graph_objects`` with a helpful error message.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RTL / Persian helpers
# ---------------------------------------------------------------------------

_RTL_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_RLM = "\u200F"  # Right-to-Left Mark
_LRM = "\u200E"  # Left-to-Right Mark


def _rtl_safe(text: str) -> str:
    """Wrap RTL (Arabic/Persian) text so matplotlib renders it correctly.

    Parameters
    ----------
    text : str
        Input string (may be Arabic, Persian, or Latin).

    Returns
    -------
    str
        Original string prefixed with RLM and suffixed with LRM if RTL
        characters are detected; otherwise the string is returned unchanged.

    Examples
    --------
    >>> _rtl_safe("age")       # Latin – unchanged
    'age'
    >>> _rtl_safe("سن")        # Persian – wrapped
    '\u200fسن\u200e'
    """
    if _RTL_PATTERN.search(str(text)):
        return f"{_RLM}{text}{_LRM}"
    return str(text)


def _safe_labels(labels: Sequence) -> List[str]:
    """Apply :func:`_rtl_safe` to every label in *labels*.

    Parameters
    ----------
    labels : sequence
        Column names, index values, or any string-like sequence.

    Returns
    -------
    list of str
    """
    return [_rtl_safe(str(lbl)) for lbl in labels]


# ---------------------------------------------------------------------------
# Nullity helpers
# ---------------------------------------------------------------------------

def _nullity(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Return a boolean DataFrame: ``True`` where data is missing.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
        Extra sentinel values treated as missing (e.g. ``[-99, "N/A"]``).

    Returns
    -------
    pd.DataFrame
        Same shape as *df*, dtype ``bool``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": [1, np.nan, -99], "b": [None, 2, 3]})
    >>> _nullity(df, missing_values=[-99])
           a      b
    0  False   True
    1   True  False
    2   True  False
    """
    if missing_values is None:
        return df.isnull()
    return df.isnull() | df.isin(missing_values)


def _pct_labels(
    df: pd.DataFrame,
    missing_values: Optional[List] = None,
) -> List[str]:
    """Return RTL-safe column labels with missingness percentage appended.

    Parameters
    ----------
    df : pd.DataFrame
    missing_values : list, optional
        Extra sentinel values.

    Returns
    -------
    list of str
        E.g. ``["age (12.5%)", "income (0.0%)"]``.
    """
    pct = _nullity(df, missing_values).mean() * 100
    return [_rtl_safe(f"{col} ({pct[col]:.1f}%)") for col in df.columns]


# ---------------------------------------------------------------------------
# Plotly lazy import
# ---------------------------------------------------------------------------

def _require_plotly():
    """Lazily import ``plotly.graph_objects``.

    Raises
    ------
    ImportError
        With an actionable message if plotly is not installed.

    Returns
    -------
    module
        ``plotly.graph_objects``.
    """
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "Interactive mode requires plotly >= 5.0. "
            "Install it with: pip install plotly"
        ) from exc
