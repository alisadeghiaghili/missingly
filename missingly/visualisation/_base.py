"""Low-level rendering helpers shared by all visualisation back-ends.

Public API
----------
_rtl_safe
    Reshape + bidi-reorder a single text label for safe matplotlib rendering.
_safe_labels
    Apply :func:`_rtl_safe` to every label in a sequence.
_apply_rtl_font
    Register a RTL-capable font with matplotlib when RTL text is detected.
_nullity
    Return a boolean DataFrame marking missing / sentinel values.
_pct_labels
    Build ``"col (X.Y%)"`` label strings for bar/vis_miss axes.
_ensure_vazirmatn
    Download / validate Vazirmatn font to local cache.
_register_font
    Register a font file with matplotlib's font manager.
"""

from __future__ import annotations

import logging
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RTL libraries -- imported at module level so tests can monkeypatch
# via ``patch.object(_base, '_ar_reshaper', mock)``.
#
# IMPORTANT: both names are ALWAYS defined at module scope (as None when the
# library is absent) so that patch.object(_base, '_ar_reshaper', ...) works
# unconditionally in test code.
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

# Alias used by newer tests that reference _base._RTL_WARNED directly.
_RTL_WARNED = _RTL_FONT_WARNED  # type: ignore[assignment]  # kept in sync via property below

# ---------------------------------------------------------------------------
# Vazirmatn font cache
# ---------------------------------------------------------------------------

_FONT_CACHE_DIR: Path = Path.home() / ".cache" / "missingly" / "fonts"

_VAZIRMATN_URL = (
    "https://github.com/rastikerdar/vazirmatn/releases/download/v33.003/"
    "Vazirmatn-Regular.ttf"
)
_VAZIRMATN_FILENAME = "Vazirmatn-Regular.ttf"
_TTF_MAGIC = b"\x00\x01\x00\x00"
_OTF_MAGIC = b"OTTO"


def _ensure_vazirmatn() -> Path | None:
    """Return a local path to a valid Vazirmatn TTF, downloading if needed.

    Returns *None* on network failure or if the downloaded data is not a
    recognised font binary.

    Parameters
    ----------
    (none)

    Returns
    -------
    Path or None
        Absolute path to ``Vazirmatn-Regular.ttf`` in the local cache, or
        *None* when the font cannot be obtained.

    Notes
    -----
    Patching :data:`_FONT_CACHE_DIR` in tests redirects the cache directory
    so that downloads go to a temporary location.
    """
    cache_dir: Path = _FONT_CACHE_DIR  # allow monkeypatching in tests
    cache_dir.mkdir(parents=True, exist_ok=True)
    font_path = cache_dir / _VAZIRMATN_FILENAME

    # Validate an existing cached file.
    if font_path.exists():
        header = font_path.read_bytes()[:4]
        if header in (_TTF_MAGIC, _OTF_MAGIC):
            return font_path
        # Corrupt / wrong file – remove and re-download.
        font_path.unlink(missing_ok=True)

    # Download.
    try:
        with urllib.request.urlopen(_VAZIRMATN_URL) as resp:
            data: bytes = resp.read()
    except OSError:
        return None

    # Validate the downloaded bytes.
    if data[:4] not in (_TTF_MAGIC, _OTF_MAGIC):
        return None

    font_path.write_bytes(data)
    return font_path


def _register_font(path: Path) -> bool:
    """Register a font file with matplotlib's font manager.

    Parameters
    ----------
    path:
        Absolute path to a ``.ttf`` or ``.otf`` font file.

    Returns
    -------
    bool
        ``True`` on success, ``False`` when matplotlib raises during
        registration (a ``WARNING``-level log entry is emitted).
    """
    try:
        fm.fontManager.addfont(str(path))
        return True
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Failed to register font %s: %s", path, exc)
        return False


def _find_rtl_font() -> str | None:
    """Return the first available RTL-capable font family name, or *None*."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _RTL_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def _apply_rtl_font(
    labels_or_ax: "Sequence | mpl.axes.Axes | None" = None,  # type: ignore[name-defined]
    ax: "mpl.axes.Axes | None" = None,  # type: ignore[name-defined]
) -> None:
    """Set a RTL-capable font on *ax* (or globally) when RTL labels detected.

    Accepts two call signatures for backward compatibility:

    * ``_apply_rtl_font(ax)``          – legacy: first arg is an Axes or None
    * ``_apply_rtl_font(labels, ax)``  – new: first arg is a label sequence

    When called with a label sequence the function is a no-op for purely
    LTR content (avoids unnecessary font switching).

    Parameters
    ----------
    labels_or_ax:
        Either a sequence of label strings **or** a
        :class:`~matplotlib.axes.Axes` instance (legacy call).  Pass
        ``None`` to apply the font globally.
    ax:
        Axes instance; only used when *labels_or_ax* is a label sequence.
    """
    global _RTL_FONT_REGISTERED, _RTL_FONT_WARNED, _RTL_WARNED

    # Normalise the two call signatures.
    if isinstance(labels_or_ax, mpl.axes.Axes):
        _ax: "mpl.axes.Axes | None" = labels_or_ax  # type: ignore[name-defined]
        labels: "Sequence | None" = None
    elif labels_or_ax is None:
        _ax = ax
        labels = None
    else:
        # labels_or_ax is a sequence of label strings.
        labels = labels_or_ax
        _ax = ax
        # Short-circuit: nothing to do for purely LTR labels.
        if not any(_is_rtl(str(lbl)) for lbl in labels):
            return

    if _RTL_FONT_REGISTERED is None:
        _RTL_FONT_REGISTERED = _find_rtl_font()

    if _RTL_FONT_REGISTERED is None:
        # Try downloading Vazirmatn as a last resort.
        font_path = _ensure_vazirmatn()
        if font_path is not None:
            _register_font(font_path)
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
            _RTL_WARNED = True
        return

    if _ax is not None:
        for item in (
            [_ax.title, _ax.xaxis.label, _ax.yaxis.label]
            + _ax.get_xticklabels()
            + _ax.get_yticklabels()
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
        return [(False, "")]

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

    _mod = sys.modules[__name__]

    for is_rtl_run, chunk in runs:
        if not is_rtl_run:
            processed_parts.append(chunk)
            continue

        # RTL segment: apply reshaper then bidi (if available).
        part = chunk

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


def _rtl_safe_labels(labels: Sequence, ax: "mpl.axes.Axes | None" = None) -> list[str]:  # type: ignore[name-defined]
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


# Public alias used by test_rtl_base.py: ``from _base import _safe_labels``
_safe_labels = _rtl_safe_labels

# Keep the old name as an alias for backward compatibility.
_rtl_safe_label_list = _rtl_safe_labels


# ---------------------------------------------------------------------------
# Nullity helpers
# ---------------------------------------------------------------------------

def _nullity(
    df: pd.DataFrame,
    missing_values: "List | None" = None,
) -> pd.DataFrame:
    """Return a boolean DataFrame where ``True`` marks a missing value.

    A cell is considered missing when it is ``NaN`` / ``None`` **or** when
    its value appears in the optional *missing_values* sentinel list.

    Parameters
    ----------
    df:
        Input DataFrame.
    missing_values:
        Additional values to treat as missing (e.g. ``[-99, "N/A", ""]``).

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with the same shape and column names as *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> _nullity(pd.DataFrame({"a": [1, np.nan]}))
       a
    0  False
    1  True
    """
    null_mask = df.isnull()

    if missing_values:
        for sentinel in missing_values:
            try:
                sentinel_mask = df.apply(
                    lambda col: col.map(lambda v: v == sentinel)
                )
                null_mask = null_mask | sentinel_mask
            except Exception:  # noqa: BLE001
                pass

    return null_mask


def _pct_labels(
    df: pd.DataFrame,
    missing_values: "List | None" = None,
) -> list[str]:
    """Build ``"col (X.Y%)"`` label strings for bar / vis_miss axes.

    Parameters
    ----------
    df:
        Input DataFrame.
    missing_values:
        Additional sentinel values forwarded to :func:`_nullity`.

    Returns
    -------
    list[str]
        One label per column, RTL-safe.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> _pct_labels(pd.DataFrame({"a": [1, np.nan, 3]}))
    ['a (33.3%)']
    """
    null_mask = _nullity(df, missing_values=missing_values)
    pcts = null_mask.mean() * 100

    labels: list[str] = []
    for col, pct in pcts.items():
        col_str = _rtl_safe(str(col))
        labels.append(f"{col_str} ({pct:.1f}%)")

    return labels


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
