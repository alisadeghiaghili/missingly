"""Performance utilities for large-dataset workflows.

.. deprecated:: 0.2.0
    All three functions were moved to ``data_quality_toolkit``.
    They remain as backwards-compatible shims and emit :class:`FutureWarning`.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Callable
import warnings

import pandas as pd

from ._deprecation import deprecated_api


@deprecated_api(
    "Use `from data_quality_toolkit.performance import memory_usage_mb` instead.",
    since="0.2.0",
)
def memory_usage_mb(*args, **kwargs):
    """Legacy shim \u2014 emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.performance``.
    """
    warnings.warn(
        "memory_usage_mb moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import memory_usage_mb as _mem
    return _mem(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.performance import optimize_dtypes` instead.",
    since="0.2.0",
)
def optimize_dtypes(*args, **kwargs):
    """Legacy shim \u2014 emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.performance``.
    """
    warnings.warn(
        "optimize_dtypes moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import optimize_dtypes as _opt
    return _opt(*args, **kwargs)


@deprecated_api(
    "Use `from data_quality_toolkit.performance import chunk_apply` instead.",
    since="0.2.0",
)
def chunk_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10_000,
    reset_index: bool = True,
):
    """Legacy shim \u2014 emits :class:`FutureWarning`.

    .. deprecated:: 0.2.0
        Moved to ``data_quality_toolkit.performance``.
    """
    warnings.warn(
        "chunk_apply moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import chunk_apply as _chunk
    return _chunk(df=df, func=func, chunk_size=chunk_size, reset_index=reset_index)
