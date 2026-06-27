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
  forms (partially broken); no extra fallback beyond bidi itself.
* **Both** installed → fully correct output.

Font fallback stack
-------------------
When an RTL font is registered, ``font.family`` is set to a **list**
(font stack) rather than a single name::

    ["Vazirmatn", "DejaVu Sans", "sans-serif"]

This allows matplotlib to fall back to DejaVu Sans for any glyph not
present in Vazirmatn – for example Greek letters (``\u03bc``) or obscure Latin
Extended characters that commonly appear in scientific column names.
Without the fallback, those glyphs render as empty boxes (\"tofu\").

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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RTL libraries – fail gracefully if not installed.
# IMPORTANT: _ar_reshaper and _bidi_get_display are ALWAYS defined at module
# level (as None when the library is absent) so that tests can use
# patch.object(_base, '_ar_reshaper', ...) unconditionally.
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

# Unicode directional markers.
_RLM = "\u200F"  # RIGHT-TO-LEFT MARK  – prepended to RTL strings
_LRM = "\u200E"  # LEFT-TO-RIGHT MARK  – appended  to RTL strings

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
# mathematical symbols like \u03bc that may be absent from Vazirmatn).
_FALLBACK_FONTS = ["DejaVu Sans", "sans-serif"]

# Direct TTF download from the official Vazirmatn GitHub repository.
_VAZIRMATN_URL = (
    "https://github.com/rastikerdar/vazirmatn/raw/master/"
    "fonts/ttf/Vazirmatn-Regular.ttf"
)
_VAZIRMATN_FONT_NAME = "Vazirmatn-Regular.ttf"
_FONT_CACHE_DIR = Path.home() / ".cache" / "missingly" / "fonts"

# Minimum size of a valid TTF file (magic bytes check).
_TTF_MAGIC = b"\x00\x01\x00\x00"
_OTF_MAGIC = b"OTTO"
_MIN_FONT_SIZE = 16


def _is_rtl(text: str) -> bool:
    """Return ``True`` when *text* contains Arabic/Persian characters."""
    return bool(_RTL_PATTERN.search(str(text)))


def _split_runs(text: str) -> list[tuple[bool, str]]:
    """Split *text* into alternating RTL/LTR runs.

    Each element is a ``(is_rtl, chunk)`` tuple.  Joining all chunks
    reproduces the original string exactly.

    Parameters
    ----------
    text : str
        Input string (may be empty or purely one script).

    Returns
    -------
    list of (bool, str)
        ``is_rtl`` is ``True`` when the chunk contains Arabic/Persian chars.

    Examples
    --------
    >>> _split_runs("age")
    [(False, 'age')]
    >>> _split_runs("\u0633\u0646 (42.5%)")
    [(True, '\u0633\u0646'), (False, ' (42.5%)')]
    """
    if not text:
        return [(False, "")]

    runs: list[tuple[bool, str]] = []
    current_is_rtl: bool = _is_rtl(text[0])
    current_chunk: list[str] = [text[0]]

    for ch in text[1:]:
        ch_is_rtl = bool(_RTL_PATTERN.match(ch))
        if ch_is_rtl == current_is_rtl:
            current_chunk.append(ch)
        else:
            runs.append((current_is_rtl, "".join(current_chunk)))
            current_is_rtl = ch_is_rtl
            current_chunk = [ch]

    runs.append((current_is_rtl, "".join(current_chunk)))
    return runs


def _rtl_safe(text: str) -> str:
    """Reshape and reorder *text* for correct matplotlib rendering.

    Processes the string **run by run** so that embedded LTR segments
    (e.g. percentages, numbers, parentheses) are never passed through the
    Arabic reshaper and are preserved exactly.

    For strings that contain any RTL characters the result is wrapped with
    a RIGHT-TO-LEFT MARK (\\u200F) prefix and a LEFT-TO-RIGHT MARK (\\u200E)
    suffix.  This signals to matplotlib's text renderer that the whole label
    is RTL and prevents the surrounding layout from mirroring the LTR parts.

    Parameters
    ----------
    text : str
        Label text, possibly containing Arabic/Persian characters.

    Returns
    -------
    str
        Processed string suitable for use as a matplotlib axis label.
        Pure LTR strings are returned unchanged.
        RTL-containing strings start with \\u200F and end with \\u200E.
    """
    text = str(text)

    # Fast-path: purely Latin/numeric – no RTL processing needed.
    if not _is_rtl(text):
        return text

    runs = _split_runs(text)

    processed_parts: list[str] = []
    for is_rtl_run, chunk in runs:
        if not is_rtl_run:
            # LTR segment: never touch it.
            processed_parts.append(chunk)
            continue

        # RTL segment: apply reshaper then bidi (if available).
        part = chunk
        if _HAVE_RESHAPER and _ar_reshaper is not None:
            try:
                part = _ar_reshaper.reshape(part)
            except Exception:  # noqa: BLE001
                pass

        if _HAVE_BIDI and _bidi_get_display is not None:
            try:
                part = _bidi_get_display(part)
            except Exception:  # noqa: BLE001
                pass
        elif not _HAVE_BIDI:
            # No bidi: reverse as last-resort order fix.
            part = part[::-1]

        processed_parts.append(part)

    # Wrap the entire result with directional markers so matplotlib's layout
    # engine treats the label as RTL regardless of which fallback path ran.
    return _RLM + "".join(processed_parts) + _LRM


def _safe_labels(labels: Sequence, missing_values: Optional[list] = None) -> list:
    """Apply :func:`_rtl_safe` to every element in *labels*.

    Parameters
    ----------
    labels : sequence
        Column names or any sequence of label values.
    missing_values : list, optional
        Unused; accepted for API symmetry with :func:`_pct_labels`.

    Returns
    -------
    list of str
    """
    return [_rtl_safe(str(lbl)) for lbl in labels]


def _ensure_vazirmatn() -> Optional[Path]:
    """Download and cache Vazirmatn-Regular.ttf if not already present.

    Validates the cached file with a TTF/OTF magic-byte check and
    re-downloads if the cached copy is corrupt.

    Returns
    -------
    pathlib.Path or None
        Path to the cached font file, or ``None`` if the download fails.
    """
    font_path = _FONT_CACHE_DIR / _VAZIRMATN_FONT_NAME

    def _is_valid_font(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < _MIN_FONT_SIZE:
            return False
        magic = path.read_bytes()[:4]
        return magic in (_TTF_MAGIC, _OTF_MAGIC)

    if font_path.exists():
        if _is_valid_font(font_path):
            return font_path
        # Corrupt cache: remove and re-download.
        try:
            font_path.unlink()
        except OSError:
            pass

    try:
        import urllib.request

        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_VAZIRMATN_URL, timeout=10) as resp:
            data = resp.read()

        # Validate before saving.
        if len(data) < _MIN_FONT_SIZE or data[:4] not in (_TTF_MAGIC, _OTF_MAGIC):
            return None

        font_path.write_bytes(data)
        return font_path

    except Exception:  # network error, permission, etc.
        return None


def _register_font(font_path: Path) -> bool:
    """Register *font_path* with matplotlib's font manager.

    Parameters
    ----------
    font_path : pathlib.Path
        Path to the TTF/OTF font file to register.

    Returns
    -------
    bool
        ``True`` if registration succeeded, ``False`` otherwise.
    """
    try:
        import matplotlib.font_manager as fm

        fm.fontManager.addfont(str(font_path))
        # Build font stack: RTL font first, then Latin fallbacks.
        font_name = Path(font_path).stem.split("-")[0]  # e.g. "Vazirmatn"
        stack = [font_name] + _FALLBACK_FONTS
        import matplotlib as mpl
        mpl.rcParams["font.family"] = stack
        return True
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to register font %s: %s", font_path, exc)
        return False


def _apply_rtl_font(labels: Sequence) -> None:
    """Configure matplotlib to use a Persian-capable font when needed.

    Called automatically by every visualisation function before rendering.
    Emits a :class:`UserWarning` **at most once per Python session** when
    RTL text is detected but no suitable font can be found or downloaded.

    Parameters
    ----------
    labels : sequence
        The axis tick labels that will be rendered.
    """
    global _RTL_WARNED, _RTL_FONT_REGISTERED  # noqa: PLW0603

    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return

    if _RTL_FONT_REGISTERED:
        return

    import matplotlib.font_manager as fm

    # Check whether a suitable system font is already available.
    available_names = {f.name for f in fm.fontManager.ttflist}
    for candidate in _RTL_FONT_CANDIDATES:
        if candidate in available_names:
            try:
                import matplotlib as mpl
                mpl.rcParams["font.family"] = [candidate] + _FALLBACK_FONTS
                _RTL_FONT_REGISTERED = True
                return
            except Exception:  # noqa: BLE001
                continue

    # No system font found — try to download Vazirmatn.
    font_path = _ensure_vazirmatn()
    if font_path is not None:
        if _register_font(font_path):
            _RTL_FONT_REGISTERED = True
            return

    # All attempts failed.
    if not _RTL_WARNED:
        warnings.warn(
            "Persian/Arabic column names detected but no suitable font is available. "
            "Install the RTL extras for correct rendering:\n"
            "    pip install missingly[rtl]\n"
            "or explicitly:\n"
            "    pip install arabic-reshaper python-bidi",
            UserWarning,
            stacklevel=3,
        )
        _RTL_WARNED = True


def _rtl_plotly_layout(labels: Sequence) -> dict:
    """Return Plotly layout kwargs for RTL-aware rendering.

    Parameters
    ----------
    labels : sequence
        Axis labels to inspect for RTL content.

    Returns
    -------
    dict
        Extra keyword arguments to merge into a Plotly ``layout`` call.
        Empty dict when no RTL text is detected.
    """
    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return {}
    return {
        "font": {
            "family": (
                "Vazirmatn, Vazir, B Nazanin, Tahoma, "
                "Arial Unicode MS, Noto Sans Arabic, sans-serif"
            )
        },
        "xaxis": {"automargin": True},
        "yaxis": {"automargin": True},
    }


# ---------------------------------------------------------------------------
# Core data helpers
# ---------------------------------------------------------------------------

def _nullity(
    df: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> pd.DataFrame:
    """Return a boolean DataFrame: ``True`` where values are missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Extra sentinel values treated as missing (e.g. ``[-99, "N/A"]``).

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with the same shape as *df*.
    """
    null_mask = df.isna()
    if missing_values:
        for val in missing_values:
            null_mask = null_mask | (df == val)
    return null_mask


def _pct_labels(
    df: pd.DataFrame,
    missing_values: Optional[list] = None,
) -> list:
    """Return column labels with missingness percentage appended.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    missing_values : list, optional
        Extra sentinel values treated as missing.

    Returns
    -------
    list of str
        Labels in the form ``"<_rtl_safe(col)> (<pct>%)"``.
    """
    null_mask = _nullity(df, missing_values=missing_values)
    pcts = null_mask.mean() * 100
    labels = []
    for col in df.columns:
        pct_str = f"{pcts[col]:.1f}%"
        safe_col = _rtl_safe(str(col))
        labels.append(f"{safe_col} ({pct_str})")
    return labels


def _require_plotly():
    """Lazy import of ``plotly.graph_objects`` with a helpful error.

    Returns
    -------
    module
        ``plotly.graph_objects``

    Raises
    ------
    ImportError
        With an install hint when plotly is not installed.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
        return go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive visualisations. "
            "Install it with: pip install plotly"
        ) from exc
