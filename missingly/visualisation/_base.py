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
    When no suitable font is found on the system, Vazirmatn is downloaded from
    GitHub and cached in ``~/.cache/missingly/fonts/`` automatically.
_rtl_plotly_layout
    Return a dict of Plotly layout kwargs that enable RTL rendering when any
    of the supplied labels contain Persian/Arabic characters.
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

Both libraries must be present for fully correct output.  The fallback
behaviour when one or both are missing is:

* **Neither** installed → character reversal (imperfect but readable).
* **Only reshaper** installed → glyphs are shaped but order is still LTR
  (partially broken); reversal applied as a best-effort order fix.
* **Only bidi** installed → order is corrected but glyphs may be isolated
  forms (partially broken); no extra fallback possible beyond bidi itself.
* **Both** installed → fully correct output.

Font fallback stack
-------------------
When an RTL font is registered, ``font.family`` is set to a **list**
(font stack) rather than a single name::

    ["Vazirmatn", "DejaVu Sans", "sans-serif"]

This allows matplotlib to fall back to DejaVu Sans for any glyph not
present in Vazirmatn – for example Greek letters (``μ``) or obscure Latin
Extended characters that commonly appear in scientific column names.
Without the fallback, those glyphs render as empty boxes ("tofu").

Font auto-download
------------------
On systems without a Persian/Arabic-capable font (common on macOS and Linux),
:func:`_apply_rtl_font` will silently download **Vazirmatn-Regular.ttf** (a
high-quality open-source Persian font by Saber Rastikerdar) directly from the
official GitHub repository and cache it at
``~/.cache/missingly/fonts/Vazirmatn-Regular.ttf``.  The download happens at
most once per machine; subsequent calls use the cached file.

If the download fails (no network, permission error, etc.) the function falls
back gracefully and emits a :class:`UserWarning` with install instructions.

A ``UserWarning`` is emitted **once per Python session** when Persian/Arabic
labels are detected but the required libraries (or a suitable system font) are
not available, so users know exactly what to install::

    pip install missingly[rtl]
    # or explicitly:
    pip install arabic-reshaper python-bidi

For Plotly figures, :func:`_rtl_plotly_layout` injects a RTL-capable font
family and sets ``automargin=True`` on axes that carry Persian/Arabic labels.
No additional libraries are required for the Plotly path.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

# Emitted at most once per session to avoid spamming the user.
_RTL_WARNED: bool = False

# Set to True once a font has been successfully registered this session.
_RTL_FONT_REGISTERED: bool = False

# ---------------------------------------------------------------------------
# RTL / Persian helpers
# ---------------------------------------------------------------------------

_RTL_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Preferred fonts for RTL rendering, ordered by quality/availability.
_RTL_FONT_CANDIDATES = [
    "Vazirmatn",
    "Vazir",
    "B Nazanin",
    "Iranian Sans",
    "Tahoma",
    "Arial Unicode MS",
    "Noto Sans Arabic",
]

# Fallback fonts for non-RTL glyphs (Greek letters, obscure Latin Extended,
# mathematical symbols like μ that may be absent from Vazirmatn).
_FALLBACK_FONTS = ["DejaVu Sans", "sans-serif"]

# Direct TTF download from the official Vazirmatn GitHub repository.
# Using the raw file URL avoids dealing with zip extraction and is more
# robust across releases.  Pinned to the master branch for latest stable.
_VAZIRMATN_URL = (
    "https://github.com/rastikerdar/vazirmatn/raw/master/"
    "fonts/ttf/Vazirmatn-Regular.ttf"
)
_VAZIRMATN_FONT_NAME = "Vazirmatn-Regular.ttf"
_FONT_CACHE_DIR = Path.home() / ".cache" / "missingly" / "fonts"


def _is_rtl(text: str) -> bool:
    """Return ``True`` when *text* contains Arabic/Persian characters."""
    return bool(_RTL_PATTERN.search(str(text)))


def _ensure_vazirmatn() -> Optional[Path]:
    """Download and cache Vazirmatn-Regular.ttf if not already present.

    Downloads the TTF file directly from the official Vazirmatn GitHub
    repository.  The file is cached at
    ``~/.cache/missingly/fonts/Vazirmatn-Regular.ttf`` and is downloaded at
    most once per machine.

    Returns
    -------
    pathlib.Path or None
        Path to the cached font file, or ``None`` if the download fails for
        any reason (no network, permission error, unexpected response, etc.).
    """
    font_path = _FONT_CACHE_DIR / _VAZIRMATN_FONT_NAME
    if font_path.exists():
        return font_path

    try:
        import urllib.request

        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_VAZIRMATN_URL, timeout=10) as resp:
            font_path.write_bytes(resp.read())
        return font_path

    except Exception:  # network error, permission, etc.
        return None


def _register_font(font_path: Path) -> bool:
    """Register *font_path* with matplotlib's font manager.

    Calls :func:`matplotlib.font_manager.fontManager.addfont` and sets
    ``matplotlib.rcParams["font.family"]`` to a **font stack** that starts
    with the registered RTL font and falls back to DejaVu Sans for any
    glyph not present in the RTL font (e.g. Greek letters such as ``μ``).

    The stack is equivalent to the CSS ``font-family`` cascade::

        ["Vazirmatn", "DejaVu Sans", "sans-serif"]

    Parameters
    ----------
    font_path : pathlib.Path
        Path to the TTF/OTF file to register.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any error.
    """
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        rtl_name = prop.get_name()
        # Build a stack: RTL font first, then generic fallbacks for non-RTL
        # glyphs (Greek, obscure Latin Extended, mathematical symbols, etc.)
        mpl.rcParams["font.family"] = [rtl_name] + _FALLBACK_FONTS
        return True
    except Exception:
        return False


def _rtl_safe(text: str) -> str:
    """Reshape and reorder RTL (Arabic/Persian) text for matplotlib.

    Applies glyph shaping via ``arabic-reshaper`` and visual reordering via
    ``python-bidi`` so that matplotlib – which has no built-in BiDi support –
    renders Persian and Arabic labels correctly.

    Fallback behaviour when libraries are missing:

    * **Both** installed → fully correct output.
    * **Only reshaper** → glyphs shaped, then string reversed for display order.
    * **Only bidi** → bidi reordering applied (glyphs may be isolated forms).
    * **Neither** → string reversed as a last-resort order fix.

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
    '\\u0646\\u0633'
    """
    s = str(text)
    if not _is_rtl(s):
        return s

    if _HAVE_RESHAPER and _HAVE_BIDI:
        s = _ar_reshaper.reshape(s)
        s = _bidi_get_display(s)
    elif _HAVE_RESHAPER:
        s = _ar_reshaper.reshape(s)
        s = s[::-1]
    elif _HAVE_BIDI:
        s = _bidi_get_display(s)
    else:
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
    attempts to configure ``font.family`` with a **font stack** that puts
    the RTL font first and falls back to DejaVu Sans for glyphs missing
    from the RTL font (e.g. Greek letters such as ``μ``).

    Resolution order
    ~~~~~~~~~~~~~~~~
    1. Scan fonts already registered with matplotlib (system fonts + any
       previously added custom fonts).  Use the first match from
       ``_RTL_FONT_CANDIDATES``.
    2. If none found, silently download **Vazirmatn-Regular.ttf** directly
       from GitHub and cache it at
       ``~/.cache/missingly/fonts/Vazirmatn-Regular.ttf``.  Register it
       with matplotlib and set it as the active font family.
    3. If the download fails (no network, permission error, etc.), fall back
       to matplotlib's default (DejaVu Sans) and emit a :class:`UserWarning`
       **once per session** with actionable install instructions.

    Parameters
    ----------
    labels : sequence
        Any iterable of strings (column names, tick labels, title, …).
    """
    global _RTL_WARNED, _RTL_FONT_REGISTERED

    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return

    import matplotlib as mpl
    import matplotlib.font_manager as fm

    # Step 1: check fonts already known to matplotlib (system + previously cached)
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _RTL_FONT_CANDIDATES:
        if font in available:
            # Set as a stack so non-RTL glyphs fall back to DejaVu Sans
            mpl.rcParams["font.family"] = [font] + _FALLBACK_FONTS
            return

    # Step 2: auto-download Vazirmatn (once per session)
    if not _RTL_FONT_REGISTERED:
        font_path = _ensure_vazirmatn()
        if font_path is not None and _register_font(font_path):
            _RTL_FONT_REGISTERED = True
            return

    # Step 3: nothing worked – warn once
    if not _RTL_WARNED:
        _RTL_WARNED = True
        warnings.warn(
            "Persian/Arabic labels detected, but missingly could not load a "
            "suitable font for matplotlib.\n\n"
            "For best results, install the RTL support libraries:\n\n"
            "    pip install missingly[rtl]\n\n"
            "These libraries (arabic-reshaper, python-bidi) reshape glyphs and "
            "apply the Unicode Bidi Algorithm so that Persian text displays "
            "correctly in static plots.\n\n"
            "Alternatively, install a Persian font such as Vazirmatn on your "
            "system, or ensure internet access so missingly can download it "
            "automatically.\n\n"
            "For interactive Plotly plots (interactive=True) no extra install "
            "is needed — they already render Persian correctly.",
            UserWarning,
            stacklevel=3,
        )


def _rtl_plotly_layout(labels: Sequence) -> Dict[str, Any]:
    """Return Plotly layout kwargs that enable correct RTL rendering.

    Plotly's WebGL/SVG renderer handles Unicode Bidi to some extent, but
    still needs an explicit RTL-capable font to display connected
    Persian/Arabic glyph forms correctly.  This helper detects whether any
    label in *labels* is RTL and, if so, returns layout overrides that:

    * Set ``font.family`` to the best available RTL-capable web-safe font.
    * Enable ``automargin`` on both axes so long RTL tick labels are not
      clipped.

    The returned dict is meant to be passed directly to
    ``fig.update_layout(**_rtl_plotly_layout(labels))``.

    Parameters
    ----------
    labels : sequence
        Any iterable of strings that will appear as axis labels or tick text
        in the figure.

    Returns
    -------
    dict
        Empty dict when no RTL text is detected (no-op).  Otherwise a dict
        with ``font``, ``xaxis``, and ``yaxis`` keys.

    Examples
    --------
    >>> fig.update_layout(**_rtl_plotly_layout(df.columns))
    """
    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return {}

    rtl_font_family = (
        "Vazirmatn, Vazir, Tahoma, Arial Unicode MS, "
        "Noto Sans Arabic, sans-serif"
    )
    return {
        "font": {"family": rtl_font_family},
        "xaxis": {"automargin": True},
        "yaxis": {"automargin": True},
    }


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
