"""Internal utilities subpackage for missingly.

This subpackage contains helpers that support the missingly core but are
not part of the public missingness-analysis API.  Public consumers should
not import from here directly — use the top-level ``missingly`` namespace.

Contents
--------
manipulation
    Data-frame manipulation utilities (replace_with_na, add_any_miss_var,
    bind_shadow_matrix).  These were previously in the root
    ``missingly/manipulation.py``; that file is now a deprecated shim.
"""

from __future__ import annotations

from .manipulation import (
    replace_with_na,
    replace_with_na_all,
    add_any_miss_var,
    bind_shadow_matrix,
)

__all__ = [
    "replace_with_na",
    "replace_with_na_all",
    "add_any_miss_var",
    "bind_shadow_matrix",
]
