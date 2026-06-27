"""Shared helpers for the visualisation layer.

This module is **private** – import from :mod:`missingly.visualisation` or
:mod:`missingly.visualise` instead.

Contents
--------
_rtl_safe
    Reshape and reorder Arabic/Persian text so matplotlib renders it correctly.
    Handles mixed RTL+LTR strings (e.g. "سن (42.5%)") by operating on
    *typed runs* — only RTL runs are passed to arabic_reshaper, preserving
    numbers, parentheses, and percent signs in their original form.
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

Both libraries must be present for fully correct output.

Mixed RTL+LTR strings
~~~~~~~~~~~~~~~~~~~~~
Labels like ``"سن (42.5%)"`` contain a Persian *run* followed by an LTR run
(space, parentheses, numbers, percent).  The old implementation fed the full
string to ``arabic_reshaper``, which corrupted the LTR run, producing labels
like ``")%5.24( نس"``.

The current implementation uses *run-aware processing*:

1. Split the input into typed runs: RTL characters vs. LTR/neutral characters.
2. Apply ``arabic_reshaper`` **only to RTL runs**.
3. Reassemble the runs into a single string.
4. Pass the reassembled string to ``python-bidi`` so the UBA handles the
   final visual ordering of the whole string correctly.

Fallback behaviour when one or both libraries are missing:

* **Both** installed → fully correct output (run-aware reshape + bidi).
* **Only reshaper** installed → RTL runs are shaped, full string reversed as
  a best-effort order fix.
* **Only bidi** installed → bidi reordering applied to raw string; glyph
  forms may be isolated (partially broken) but order is correct.
* **Neither** installed → string reversed as last-resort order fix.

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

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RTL libraries – fail gracefully if not installed
#
# _ar_reshaper and _bidi_get_display are always defined at module scope so
# that tests can monkeypatch them with patch.object(_base, "_ar_reshaper", ...)
# regardless of whether the underlying libraries are installed.  When a
# library is absent the name is set to None; _rtl_safe checks _HAVE_RESHAPER /
# _HAVE_BIDI before calling through these names.
# ---------------------------------------------------------------------------

try:
    import arabic_reshaper as _ar_reshaper  # type: ignore
    _HAVE_RESHAPER = True
except ImportError:
    _ar_reshaper = None  # type: ignore[assignment]
    _HAVE_RESHAPER = False

try:
    from bidi.algorithm import get_display as _bidi_get_display  # type: ignore
    _HAVE_BIDI = True
except ImportError:
    _bidi_get_display = None  # type: ignore[assignment]
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

# Valid TTF/OTF magic bytes for font file validation.
_FONT_MAGIC = (
    b"\x00\x01\x00\x00",  # TrueType
    b"OTTO",              # OpenType CFF
    b"true",              # Apple TrueType
    b"typ1",              # PostScript Type 1 in sfnt
)


def _is_rtl(text: str) -> bool:
    """Return ``True`` when *text* contains Arabic/Persian characters."""
    return bool(_RTL_PATTERN.search(str(text)))


def _split_runs(text: str) -> List[Tuple[bool, str]]:
    """Split *text* into alternating RTL and LTR/neutral character runs.

    Each element in the returned list is a ``(is_rtl, substring)`` pair.
    Adjacent characters of the same type are merged into a single run.

    Parameters
    ----------
    text : str
        Input string, possibly mixing Arabic/Persian and Latin/numeric chars.

    Returns
    -------
    list of (bool, str)
        ``True`` marks an RTL run; ``False`` marks an LTR/neutral run.

    Examples
    --------
    >>> _split_runs("سن (42.5%)")
    [(True, 'سن'), (False, ' (42.5%)')]
    >>> _split_runs("age")
    [(False, 'age')]
    >>> _split_runs("col_سن_val")
    [(False, 'col_'), (True, 'سن'), (False, '_val')]
    """
    if not text:
        return [(False, "")]

    runs: List[Tuple[bool, str]] = []
    current_is_rtl = _is_rtl(text[0])
    current_chars: List[str] = [text[0]]

    for ch in text[1:]:
        ch_is_rtl = _is_rtl(ch)
        if ch_is_rtl == current_is_rtl:
            current_chars.append(ch)
        else:
            runs.append((current_is_rtl, "".join(current_chars)))
            current_is_rtl = ch_is_rtl
            current_chars = [ch]

    runs.append((current_is_rtl, "".join(current_chars)))
    return runs


def _ensure_vazirmatn() -> Optional[Path]:
    """Download and cache Vazirmatn-Regular.ttf if not already present.

    Downloads the TTF file directly from the official Vazirmatn GitHub
    repository.  The file is cached at
    ``~/.cache/missingly/fonts/Vazirmatn-Regular.ttf`` and is downloaded at
    most once per machine.

    The downloaded file is validated by checking its magic bytes to guard
    against corrupt or partial downloads.  An invalid file is removed and
    ``None`` is returned so the caller can fall back gracefully.

    Returns
    -------
    pathlib.Path or None
        Path to the cached font file, or ``None`` if the download fails or
        produces an invalid file.
    """
    font_path = _FONT_CACHE_DIR / _VAZIRMATN_FONT_NAME

    # Validate existing cache – a previous download may have been corrupted.
    if font_path.exists():
        try:
            header = font_path.read_bytes()[:4]
            if any(header == magic for magic in _FONT_MAGIC):
                return font_path
            # File exists but is not a valid font – remove and re-download.
            _log.warning(
                "Cached Vazirmatn font at %s appears corrupt (bad magic "
                "bytes: %r). Removing and re-downloading.", font_path, header
            )
            font_path.unlink(missing_ok=True)
        except OSError as exc:
            _log.warning("Could not read cached font file %s: %s", font_path, exc)
            return None

    try:
        import urllib.request

        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_VAZIRMATN_URL, timeout=10) as resp:
            data = resp.read()

        # Validate before writing.
        if not any(data[:4] == magic for magic in _FONT_MAGIC):
            _log.warning(
                "Downloaded Vazirmatn font has unexpected magic bytes %r; "
                "skipping cache write.", data[:4]
            )
            return None

        font_path.write_bytes(data)
        return font_path

    except Exception as exc:
        _log.warning("Could not download Vazirmatn font: %s", exc)
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
    except Exception as exc:
        _log.warning(
            "Failed to register font %s with matplotlib: %s. "
            "Persian labels may render incorrectly.", font_path, exc
        )
        return False


def _rtl_safe(text: str) -> str:
    """Reshape and reorder RTL (Arabic/Persian) text for matplotlib.

    Handles **mixed RTL+LTR strings** correctly by operating on *typed runs*
    rather than the full string.  For example, ``"سن (42.5%)"`` contains an
    RTL run (``"سن"``) and an LTR run (``" (42.5%)"``).  Only the RTL run is
    passed to ``arabic_reshaper``; the LTR run is left untouched.  The
    reassembled string is then passed to ``python-bidi`` for final visual
    reordering.

    This prevents the corruption that occurred in the previous implementation
    where feeding the full string to ``arabic_reshaper`` mangled numbers and
    punctuation, producing labels like ``")%5.24( نس"``.

    Fallback behaviour when libraries are missing:

    * **Both** installed → fully correct output (run-aware reshape + bidi).
    * **Only reshaper** installed → RTL runs shaped, full string reversed as
      best-effort order fix.
    * **Only bidi** installed → bidi reordering applied to raw string (glyph
      forms may be isolated; order is correct).
    * **Neither** installed → full string reversed as last-resort order fix.

    Latin-only text is returned unchanged without any processing.

    Parameters
    ----------
    text : str
        Input string (may be Arabic/Persian, Latin, or mixed).

    Returns
    -------
    str
        Visually-ordered, glyph-shaped string suitable for matplotlib text
        calls; unchanged for Latin-only input.

    Examples
    --------
    >>> _rtl_safe("age")          # Latin only – unchanged
    'age'
    >>> _rtl_safe("سن")           # Pure Persian – reshaped and reordered
    'نس'
    >>> _rtl_safe("سن (42.5%)")   # Mixed – Persian part shaped, LTR preserved
    '(42.5%) نس'
    """
    s = str(text)
    if not _is_rtl(s):
        return s

    if _HAVE_RESHAPER and _HAVE_BIDI:
        # Step 1: split into typed runs
        runs = _split_runs(s)
        # Step 2: reshape only RTL runs (use module-level _ar_reshaper so
        # tests can monkeypatch it)
        reshaped_parts = [
            _ar_reshaper.reshape(chunk) if is_rtl else chunk  # type: ignore[union-attr]
            for is_rtl, chunk in runs
        ]
        # Step 3: reassemble and apply bidi for final visual ordering
        return _bidi_get_display("".join(reshaped_parts))  # type: ignore[misc]

    if _HAVE_RESHAPER:
        # Shape RTL runs only, then reverse the whole string as order fix.
        runs = _split_runs(s)
        reshaped_parts = [
            _ar_reshaper.reshape(chunk) if is_rtl else chunk  # type: ignore[union-attr]
            for is_rtl, chunk in runs
        ]
        return "".join(reshaped_parts)[::-1]

    if _HAVE_BIDI:
        # No shaping available; bidi handles ordering (glyphs may be isolated).
        return _bidi_get_display(s)  # type: ignore[misc]

    # Last resort: reverse the whole string.
    return s[::-1]


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
        E.g. ``["age (12.5%)", "income (0.0%)"]`` for Latin columns,
        or ``["سن (12.5%)", "درآمد (0.0%)"]`` for Persian columns
        (correctly shaped and reordered for matplotlib).
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
