"""missingly.utils — out-of-scope helpers isolated from the core package.

This subpackage contains utilities that were historically part of
``missingly`` but fall outside its core charter (missingness analysis,
imputation, and reporting).  They remain here for backward compatibility
and will be redirected to their canonical homes in future releases.

Submodules
----------
manipulation
    DataFrame manipulation helpers: ``replace_with_na``,
    ``replace_with_na_all``, ``add_any_miss_var``, ``bind_shadow_matrix``.
    The deprecated functions ``clean_names``, ``remove_empty``,
    ``coalesce_columns``, and ``miss_as_feature`` live here as shims
    pointing to ``data_quality_toolkit``.

Public re-exports
-----------------
All public symbols from ``missingly.utils.manipulation`` are re-exported
here so callers can import from either location::

    from missingly.utils import replace_with_na
    from missingly.utils.manipulation import replace_with_na  # identical
"""

from __future__ import annotations

from missingly.utils.manipulation import (
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
