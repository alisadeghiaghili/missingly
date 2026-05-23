"""Deprecation helpers for missingly.

Provides :func:`deprecated_api`, a decorator that emits a
:class:`FutureWarning` when a legacy or experimental API is called.

Example
-------
>>> from missingly._deprecation import deprecated_api
>>> @deprecated_api("use `new_function` instead", since="0.2.0")
... def old_function():
...     pass
"""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Optional


def deprecated_api(
    message: Optional[str] = None,
    *,
    since: Optional[str] = None,
    stacklevel: int = 2,
) -> Callable:
    """Decorator that emits a :class:`FutureWarning` on every call.

    Parameters
    ----------
    message : str, optional
        Extra hint appended to the standard warning text.
    since : str, optional
        Version string when the deprecation was introduced (e.g. ``"0.2.0"``).
    stacklevel : int, default 2
        Passed to :func:`warnings.warn`.

    Returns
    -------
    Callable
        The decorated function, with ``__wrapped__`` set to the original.

    Examples
    --------
    >>> @deprecated_api("Use `new_fn` instead.", since="0.2.0")
    ... def legacy_fn(x):
    ...     return x * 2
    """
    def decorator(func: Callable) -> Callable:
        since_str = f" (since v{since})" if since else ""
        extra = f"  {message}" if message else ""
        warn_msg = (
            f"`{func.__qualname__}` is part of the experimental / legacy API"
            f"{since_str} and may be moved, renamed, or removed in a future "
            f"release without a major-version bump.{extra}"
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(warn_msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
        return wrapper

    return decorator
