"""Performance utilities for large-dataset workflows.

This module provides three complementary tools that make missingly
usable on DataFrames that do not fit comfortably in memory or that
take too long to process in a single pass::

    - chunk_apply
    - memory_usage_mb
    - optimize_dtypes

In v0.2 these functions were moved to the separate
``data_quality_toolkit`` package. They remain available here as
backwards-compatible shims for one release cycle.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Callable
import warnings

import pandas as pd


def memory_usage_mb(*args, **kwargs):
    """Deprecated shim for :func:`data_quality_toolkit.performance.memory_usage_mb`.

    Use ``data_quality_toolkit.performance.memory_usage_mb`` instead.
    """
    warnings.warn(
        "memory_usage_mb moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import memory_usage_mb as _mem
    return _mem(*args, **kwargs)


def optimize_dtypes(*args, **kwargs):
    """Deprecated shim for :func:`data_quality_toolkit.performance.optimize_dtypes`.

    Use ``data_quality_toolkit.performance.optimize_dtypes`` instead.
    """
    warnings.warn(
        "optimize_dtypes moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import optimize_dtypes as _opt
    return _opt(*args, **kwargs)


def chunk_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10_000,
    reset_index: bool = True,
):
    """Deprecated shim for :func:`data_quality_toolkit.performance.chunk_apply`.

    The behaviour is identical; this wrapper only exists for backwards
    compatibility and will be removed in a future release.
    """
    warnings.warn(
        "chunk_apply moved to data_quality_toolkit.performance and will "
        "be removed from missingly in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    from data_quality_toolkit.performance import chunk_apply as _chunk
    return _chunk(df=df, func=func, chunk_size=chunk_size, reset_index=reset_index)
