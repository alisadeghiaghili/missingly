"""Backward-compatibility shim — summary functions live in missingly.diagnostics.

This module was merged into ``missingly.diagnostics`` in v0.3.0.
It is kept as a deprecated re-export shim so that code written against
pre-v0.3 imports continues to work until v0.5.0.

.. deprecated:: 0.3.0
    Import directly from :mod:`missingly.diagnostics` instead::

        from missingly.diagnostics import n_miss, n_complete, pct_miss, pct_complete
from missingly.diagnostics import miss_var_summary, miss_case_summary, summarise
"""
from __future__ import annotations

import warnings

warnings.warn(
    "missingly.summary is deprecated and will be removed in v0.5.0. "
    "Import from missingly.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from missingly.diagnostics import (  # noqa: E402, F401
    bind_shadow,
    n_miss,
    n_complete,
    pct_miss,
    pct_complete,
    miss_var_summary,
    miss_case_summary,
    summarise,
    miss_scan_count,
    miss_summary,
)

__all__ = [
    "bind_shadow",
    "n_miss",
    "n_complete",
    "pct_miss",
    "pct_complete",
    "miss_var_summary",
    "miss_case_summary",
    "summarise",
    "miss_scan_count",
    "miss_summary",
]
