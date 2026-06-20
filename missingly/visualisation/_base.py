"""Shared helpers for the visualisation layer.

This module is **private** – import from :mod:`missingly.visualisation` or
:mod:`missingly.visualise` instead.

Contents
--------
_rtl_safe
    Reshape and reorder Arabic/Persian text so matplotlib renders it correctly.
_safe_labels
    Apply :func:`_rtl_safe` to every label in a sequence.
_apply_rtl_font
    Configure matplotlib to use a Persian-capable font when RTL text is present.
_nullity
    Return a boolean DataFrame marking missing positions.
_pct_labels
    Column labels with missingness percentage appended.
_require_plotly
    Lazy import of ``plotly.graph_objects`` with a helpful error message.

RTL rendering notes
-------------------
matplotlib does **not** implement the Unicode Bidirectional Algorithm (UBA) nor
OpenType glyph shaping.  Naively passing a raw Persian/Arabic string produces
two problems:

1. **Wrong glyph shapes** – Arabic letters have context-dependent forms
   (initial, medial, final, isolated); matplotlib picks isolated forms only.
2. **Wrong display order** – characters are drawn left-to-right instead of
   right-to-left.

The fix uses two optional libraries:

* ``arabic-reshaper`` – applies glyph-shaping rules so connected forms appear.
* ``python-bidi`` (``bidi.algorithm``) – applies the UBA so the visual order
  is right-to-left.

If either library is absent a best-effort fallback is used (character reversal)
which is better than raw isolated glyphs.  Install both for correct output::

    pip install arabic-reshaper python-bidi
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional RTL libraries – fail gracefully if not installed
# ---------------------------------------------------------------------------

try:
    import arabic_reshaper as _ar_reshaper  # type: ignore
    _HAVE_RESHAPER = True
except ImportError:
    _HAVE_RESHAPER = False

try:
    from bidi.algorithm import get_display as _bidi_get_display  # type: ignore
    _HAVE_BIDI = True
except ImportError:
    _HAVE_BIDI = False


# ---------------------------------------------------------------------------
# RTL / Persian helpers
# ---------------------------------------------------------------------------

_RTL_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _is_rtl(text: str) -> bool:
    """Return ``True`` when *text* contains Arabic/Persian characters."""
    return bool(_RTL_PATTERN.search(str(text)))


def _rtl_safe(text: str) -> str:
    """Reshape and reorder RTL (Arabic/Persian) text for matplotlib.

    Applies glyph shaping via ``arabic-reshaper`` and visual reordering via
    ``python-bidi`` so that matplotlib – which has no built-in BiDi support –
    renders Persian and Arabic labels correctly.

    If the optional libraries are not installed a fallback is applied:
    the string is reversed character-by-character, which is imperfect but
    still far better than the default left-to-right isolated-glyph rendering.

    Latin text is returned unchanged.

    Parameters
    ----------
    text : str
        Input string (may be Arabic, Persian, or Latin).

    Returns
    -------
    str
        Visually-ordered, glyph-shaped string suitable for matplotlib text
        calls; unchanged for Latin input.

    Examples
    --------
    >>> _rtl_safe("age")   # Latin – unchanged
    'age'
    >>> _rtl_safe("سن")   # Persian – reshaped and reordered
    '\u0646\u0633'
    """
    s = str(text)
    if not _is_rtl(s):
        return s

    # Step 1 – glyph shaping
    if _HAVE_RESHAPER:
        s = _ar_reshaper.reshape(s)
    # Step 2 – visual (display) order
    if _HAVE_BIDI:
        s = _bidi_get_display(s)
    elif not _HAVE_RESHAPER:
        # last-resort fallback: reverse the whole string
        s = s[::-1]

    return s


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


def _apply_rtl_font(labels: Sequence) -> None:
    """Set a Persian/Arabic-capable font in matplotlib rcParams if needed.

    Checks whether any of *labels* contain RTL characters and, if so,
    attempts to configure ``font.family`` with a font that supports the
    Arabic/Persian Unicode range.  Common fonts are tried in order of
    preference; matplotlib's default (DejaVu Sans) is kept as the final
    fallback.

    This function is a **side-effect helper** – call it once before drawing
    a figure that may contain RTL labels.  It modifies ``matplotlib.rcParams``
    temporarily; callers that need per-Axes control should restore rcParams
    afterwards.

    Parameters
    ----------
    labels : sequence
        Any iterable of strings (column names, tick labels, title, …).
    """
    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return

    import matplotlib as mpl
    import matplotlib.font_manager as fm

    _RTL_FONT_CANDIDATES = [
        "Vazirmatn",
        "Vazir",
        "B Nazanin",
        "Iranian Sans",
        "Tahoma",
        "Arial Unicode MS",
        "Noto Sans Arabic",
        "DejaVu Sans",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in _RTL_FONT_CANDIDATES:
        if font in available:
            mpl.rcParams["font.family"] = font
            return
    # No candidate found – DejaVu Sans is already the default, nothing to do.


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
