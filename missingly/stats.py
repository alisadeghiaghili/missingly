"""Backward-compatibility shim — stats functions live in missingly.diagnostics.

This module was merged into ``missingly.diagnostics`` in v0.3.0.
It is kept as a deprecated re-export shim so that code written against
pre-v0.3 imports continues to work until v0.5.0.

.. deprecated:: 0.3.0
    Import directly from :mod:`missingly.diagnostics` instead::

        from missingly.diagnostics import mcar_test, mar_mnar_test, diagnose_missing
"""
from __future__ import annotations

import warnings

warnings.warn(
    "missingly.stats is deprecated and will be removed in v0.5.0. "
    "Import from missingly.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from missingly.diagnostics import (  # noqa: E402, F401
    mcar_test,
    mar_mnar_test,
    diagnose_missing,
)

__all__ = ["mcar_test", "mar_mnar_test", "diagnose_missing"]
