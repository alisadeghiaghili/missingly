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
except ImportEr