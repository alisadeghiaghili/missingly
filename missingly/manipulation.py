"""Deprecated shim — manipulation logic moved to ``missingly.utils.manipulation``.

This module is kept for backward compatibility only.  All symbols are
re-exported from :mod:`missingly.utils.manipulation`.  Import from there
directly for new code::

    # Preferred (new code)
    from missingly.utils.manipulation import replace_with_na

    # Still works but will emit DeprecationWarning in v0.4.0
    from missingly.manipulation import replace_with_na

.. deprecated:: 0.3.0
    This module will be removed in v0.4.0.  Migrate imports to
    ``missingly.utils.manipulation``.
"""

from __future__ import annotations

# Re-export everything from the canonical location so existing imports
# continue to work without modification.
from missingly.utils.manipulation import (  # noqa: F401
    replace_with_na,
    replace_with_na_all,
    add_any_miss_var,
    bind_shadow_matrix,
    clean_names,
    remove_empty,
    coalesce_columns,
    miss_as_feature,
)

__all__ = [
    "replace_with_na",
    "replace_with_na_all",
    "add_any_miss_var",
    "bind_shadow_matrix",
    "clean_names",
    "remove_empty",
    "coalesce_columns",
    "miss_as_feature",
]
