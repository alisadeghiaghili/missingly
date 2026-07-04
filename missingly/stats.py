"""Deprecated — all content moved to :mod:`missingly.diagnostics`.

.. deprecated:: 0.3.0
    Import directly from ``missingly.diagnostics`` or the top-level
    ``missingly`` namespace instead.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str):
    warnings.warn(
        f"Importing {name!r} from 'missingly.stats' is deprecated and will be "
        "removed in a future release. "
        f"Use 'from missingly.diagnostics import {name}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from missingly import diagnostics
    return getattr(diagnostics, name)


# Keep direct star-imports working during the deprecation window
from missingly.diagnostics import (  # noqa: F401 E402
    mcar_test,
    mar_mnar_test,
    diagnose_missing,
)
