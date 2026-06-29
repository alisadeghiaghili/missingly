"""Low-level rendering helpers shared by all visualisation back-ends.

Public API
----------
_rtl_safe
    Reshape + bidi-reorder a single text label for safe matplotlib rendering.
    Apply :func:`_rtl_safe` to every label in a sequence.

_apply_rtl_font
    Register a RTL-capable font with matplotlib when RTL text is detected.
"""

from __future__ import annotations

import unicodedata
from typing import Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Optional RTL libraries -- imported at module level so tests can monkeypatch
# via ``patch.object(_base, '_ar_reshaper', mock)``.
#
# IMPORTANT: _ar_reshaper and _bidi_get_display are ALWAYS defined at module
# scope (as None when the library is absent) so that test code can use
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

# ---------------------------------------------------------------------------
# Directional marker characters
# ---------------------------------------------------------------------------
_RLM = "\u200F"  # RIGHT-TO-LEFT MARK  - prepended to RTL strings
_LRM = "\u200E"  # LEFT-TO-RIGHT MARK  - appended  to RTL strings

# Unicode bidirectional categories that indicate a RTL character
_RTL_BIDI_CLASSES = frozenset({
    "R",    # Right-to-Left
    "AL",   # Arabic Letter
    "RLE",  # Right-to-Left Embedding
    "RLO",  # Right-to-Left Override
    "RLI",  # Right-to-Left Isolate
})

# ---------------------------------------------------------------------------
# RTL font management
# ---------------------------------------------------------------------------

# Fonts that are known to support Persian/Arabic + Latin glyphs well.
_RTL_FONT_CANDIDATES = [
    "Vazirmatn",
    "Vazir",
    "Iran Sans",
    "IranSans",
    "Noto Naskh Arabic",
    "Noto Sans Arabic",
    "Noto Kufi Arabic",
    "Scheherazade New",
    "Amiri",
    "Tahoma",   # Windows – decent Arabic support
    "Arial",    # fallback – partial Arabic support
]

_RTL_FONT_REGISTERED: str | None = None   # cached result
_RTL_FONT_WARNED = False                   # emit warning at most once


def _find_rtl_font() -> str | None:
    """Return the first available RTL-capable font family name, or *None*."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _RTL_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def _apply_rtl_font(ax: mpl.axes.Axes | None = None) -> None:  # type: ignore[name-defined]
    """Set a RTL-capable font on *ax* (or globally when *ax* is None).

    Called automatically by :func:`_rtl_safe_labels` whenever RTL text is
    detected.  The lookup is cached after the first successful match.

    Parameters
    ----------
    ax:
        The :class:`~matplotlib.axes.Axes` instance to configure.  When
        *None* the font is applied to the global ``rcParams``.
    """
    global _RTL_FONT_REGISTERED, _RTL_FONT_WARNED

    if _RTL_FONT_REGISTERED is None:
        _RTL_FONT_REGISTERED = _find_rtl_font()

    if _RTL_FONT_REGISTERED is None:
        if not _RTL_FONT_WARNED:
            import warnings
            warnings.warn(
                "No RTL-capable font found in matplotlib's font list.  "
                "Persian/Arabic labels may render as boxes.  Install "
                "'Vazirmatn' (pip install missingly[rtl]) or another "
                "Arabic-capable font.",
                UserWarning,
                stacklevel=3,
            )
            _RTL_FONT_WARNED = True
        return

    font_props = {"family": _RTL_FONT_REGISTERED}
    if ax is not None:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontfamily(_RTL_FONT_REGISTERED)
    else:
        mpl.rcParams["font.family"] = _RTL_FONT_REGISTERED


# ---------------------------------------------------------------------------
# RTL character detection
# ---------------------------------------------------------------------------

def _is_rtl(text: str) -> bool:
    """Return *True* if *text* contains at least one RTL Unicode character.

    Parameters
    ----------
    text:
        Input string to inspect.

    Returns
    -------
    bool
        ``True`` when any character in *text* belongs to a right-to-left
        Unicode bidirectional category.

    Examples
    --------
    >>> _is_rtl("hello")     # Latin-only
    False
    >>> _is_rtl("\u0633\u0644\u0627\u0645")  # Arabic/Persian
    True
    >>> _is_rtl("age (\u0633\u0646)")        # mixed
    True
    """
    return any(unicodedata.bidirectional(ch) in _RTL_BIDI_CLASSES for ch in text)


# ---------------------------------------------------------------------------
# Run detection for mixed LTR/RTL text
# ---------------------------------------------------------------------------

def _split_runs(text: str) -> list[tuple[bool, str]]:
    """Split *text* into alternating RTL / LTR runs.

    Parameters
    ----------
    text:
        Input string, potentially containing both RTL and LTR characters.

    Returns
    -------
    list of (is_rtl, chunk)
        Each tuple contains a boolean flag and the corresponding substring.

    Examples
    --------
    >>> _split_runs("age (\u0633\u0646)")
    [(False, 'age ('), (True, '\u0633\u0646'), (False, ')')]
    """
    if not text:
        return []

    runs: list[tuple[bool, str]] = []
    current_rtl = unicodedata.bidirectional(text[0]) in _RTL_BIDI_CLASSES
    buf = text[0]

    for ch in text[1:]:
        ch_rtl = unicodedata.bidirectional(ch) in _RTL_BIDI_CLASSES
        if ch_rtl == current_rtl:
            buf += ch
        else:
            runs.append((current_rtl, buf))
            current_rtl = ch_rtl
            buf = ch

    runs.append((current_rtl, buf))
    return runs


# ---------------------------------------------------------------------------
# Core RTL-safe label processing
# ---------------------------------------------------------------------------

def _rtl_safe(text: str) -> str:
    """Return a matplotlib-safe version of *text*.

    For purely LTR text the string is returned unchanged.  For text that
    contains RTL characters the function:

    1. Splits the string into RTL / LTR runs.
    2. Applies ``arabic_reshaper.reshape`` to each RTL run (when available).
    3. Applies ``bidi.algorithm.get_display`` to each RTL run (when
       available) or reverses the run as a last-resort fallback.
    4. Wraps the full result in RLM / LRM directional markers.

    Parameters
    ----------
    text:
        Raw label string.

    Returns
    -------
    str
        Processed label ready for matplotlib rendering.

    Notes
    -----
    The function reads ``_ar_reshaper``, ``_bidi_get_display``,
    ``_HAVE_RESHAPER``, and ``_HAVE_BIDI`` through
    ``sys.modules[__name__]`` rather than as free variables so that test
    code can patch them via ``unittest.mock.patch.object``.

    Examples
    --------
    >>> result = _rtl_safe("hello")   # LTR unchanged
    >>> result == "hello"
    True
    >>> rtl = _rtl_safe("\u0633\u0644\u0627\u0645")  # RTL wrapped
    >>> rtl.startswith("\u200f")
    True
    """
    text = str(text)
    if not _is_rtl(text):
        return text

    runs = _split_runs(text)
    processed_parts: list[str] = []

    for is_rtl_run, chunk in runs:
        if not is_rtl_run:
            processed_parts.append(chunk)
            continue

        # RTL segment: apply reshaper then bidi (if available).
        part = chunk
        import sys as _sys
        _mod = _sys.modules[__name__]
        if _mod._HAVE_RESHAPER and _mod._ar_reshaper is not None:
            try:
                part = _mod._ar_reshaper.reshape(part)
            except Exception:  # noqa: BLE001
                pass

        if _mod._HAVE_BIDI and _mod._bidi_get_display is not None:
            try:
                part = _mod._bidi_get_display(part)
            except Exception:  # noqa: BLE001
                pass
        elif not _mod._HAVE_BIDI:
            # No bidi: reverse as last-resort order fix.
            part = part[::-1]

        processed_parts.append(part)

    # Wrap the entire result with directional markers so matplotlib's layout
    # engine treats the label as RTL from the very first character.
    return _RLM + "".join(processed_parts) + _LRM


def _rtl_safe_labels(labels: Sequence, ax: mpl.axes.Axes | None = None) -> list[str]:  # type: ignore[name-defined]
    """Apply :func:`_rtl_safe` to every element in *labels*.

    When RTL text is detected the function also registers a RTL-capable
    font on *ax* (or globally) via :func:`_apply_rtl_font`.

    Parameters
    ----------
    labels:
        Sequence of raw label strings (any type -- converted via ``str``).
    ax:
        Optional axes instance for font scoping.

    Returns
    -------
    list[str]
        Processed labels ready for matplotlib rendering.
    """
    safe = [_rtl_safe(str(lbl)) for lbl in labels]
    if any(s.startswith(_RLM) for s in safe):
        _apply_rtl_font(ax)
    return safe


# Keep the old name as an alias for backward compatibility.
_rtl_safe_label_list = _rtl_safe_labels


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _rtl_plotly_layout(labels: Sequence) -> dict:
    """Return a Plotly layout dict fragment for RTL support.

    When *labels* contain RTL text the fragment sets
    ``font.family`` to the first available RTL-capable font and
    configures right-to-left axis directions.

    Parameters
    ----------
    labels:
        Column / axis labels to inspect.

    Returns
    -------
    dict
        Plotly layout kwargs.  Empty dict when no RTL text is detected.
    """
    if not any(_is_rtl(str(lbl)) for lbl in labels):
        return {}

    global _RTL_FONT_REGISTERED
    if _RTL_FONT_REGISTERED is None:
        _RTL_FONT_REGISTERED = _find_rtl_font()

    font_family = _RTL_FONT_REGISTERED or "Arial"
    return {
        "font": {"family": font_family},
        "xaxis": {"autorange": "reversed"},
        "yaxis": {"autorange": "reversed"},
    }
