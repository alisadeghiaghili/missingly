"""Shared input validators for the missingly package.

All validators raise a specific :mod:`missingly.exceptions` exception on
failure.  They never return a value — callers that pass validation simply
continue execution.

Design rules
------------
* Validators are pure functions with no side-effects.
* Every error message names the offending parameter, states the
  expected constraint, and shows a concrete valid example where useful.
* Validators do **not** mutate their inputs.

Examples
--------
>>> import pandas as pd
>>> from missingly._validation import validate_dataframe, validate_columns
>>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
>>> validate_dataframe(df)          # passes silently
>>> validate_columns(df, ["a"])     # passes silently
>>> validate_columns(df, ["z"])     # raises MissingColumnError
Traceback (most recent call last):
    ...
missingly.exceptions.MissingColumnError: Column(s) not found in DataFrame: ['z']. Available columns: ['a', 'b']
"""

from __future__ import annotations

from typing import Any, Collection, List, Optional

import numpy as np
import pandas as pd

from missingly.exceptions import (
    ConfigurationError,
    InvalidStrategyError,
    MissingColumnError,
    MissinglyError,
)


def validate_dataframe(
    df: Any,
    *,
    param: str = "df",
    allow_empty: bool = False,
    min_rows: Optional[int] = None,
) -> None:
    """Assert that *df* is a non-empty pandas DataFrame.

    Parameters
    ----------
    df : Any
        Object to validate.
    param : str, default ``'df'``
        Parameter name used in error messages.
    allow_empty : bool, default False
        If True, a DataFrame with zero rows is accepted.
    min_rows : int, optional
        When given, raises if the DataFrame has fewer than *min_rows* rows.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If the DataFrame is empty (when *allow_empty* is False) or has
        fewer rows than *min_rows*.

    Examples
    --------
    >>> import pandas as pd
    >>> from missingly._validation import validate_dataframe
    >>> validate_dataframe(pd.DataFrame({"a": [1]}))
    >>> validate_dataframe("not a df")
    Traceback (most recent call last):
        ...
    TypeError: `df` must be a pandas DataFrame; got str.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"`{param}` must be a pandas DataFrame; got {type(df).__name__}."
        )
    if not allow_empty and df.empty:
        raise ValueError(
            f"`{param}` must not be empty. "
            "Pass a DataFrame with at least one row and one column."
        )
    if min_rows is not None and len(df) < min_rows:
        raise ValueError(
            f"`{param}` must have at least {min_rows} row(s); got {len(df)}."
        )


def validate_columns(
    df: pd.DataFrame,
    columns: Collection[str],
    *,
    param: str = "columns",
    require_numeric: bool = False,
) -> None:
    """Assert that all *columns* exist in *df* and optionally are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check against.
    columns : collection of str
        Column names that must be present.
    param : str, default ``'columns'``
        Parameter name used in error messages.
    require_numeric : bool, default False
        If True, also raises if any column is not numeric.

    Raises
    ------
    MissingColumnError
        If any column in *columns* is not present in *df*.
    TypeError
        If *require_numeric* is True and a column has a non-numeric dtype.

    Examples
    --------
    >>> import pandas as pd
    >>> from missingly._validation import validate_columns
    >>> df = pd.DataFrame({"age": [1.0], "city": ["Berlin"]})
    >>> validate_columns(df, ["age"])                        # passes
    >>> validate_columns(df, ["age"], require_numeric=True)  # passes
    >>> validate_columns(df, ["city"], require_numeric=True)
    Traceback (most recent call last):
        ...
    TypeError: Column 'city' must be numeric; got dtype object.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MissingColumnError(
            columns=missing,
            available=list(df.columns),
        )
    if require_numeric:
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"Column {col!r} must be numeric; got dtype {df[col].dtype}."
                )


def validate_strategy(
    strategy: Any,
    allowed: Collection[str],
    *,
    param: str = "strategy",
    example: Optional[str] = None,
) -> None:
    """Assert that *strategy* is one of the *allowed* values.

    Parameters
    ----------
    strategy : Any
        Value to validate.
    allowed : collection of str
        Accepted strategy names.
    param : str, default ``'strategy'``
        Parameter name used in error messages.
    example : str, optional
        A ready-to-use example value shown in the error message.

    Raises
    ------
    InvalidStrategyError
        If *strategy* is not in *allowed*.

    Examples
    --------
    >>> from missingly._validation import validate_strategy
    >>> validate_strategy("mean", ["mean", "median"])  # passes
    >>> validate_strategy("avg", ["mean", "median"])
    Traceback (most recent call last):
        ...
    missingly.exceptions.InvalidStrategyError: `strategy` must be one of ['mean', 'median']; got 'avg'.
    """
    if strategy not in allowed:
        raise InvalidStrategyError(
            param=param,
            got=strategy,
            allowed=sorted(allowed),
            example=example,
        )


def validate_positive_int(
    value: Any,
    *,
    param: str,
    allow_zero: bool = False,
) -> None:
    """Assert that *value* is a positive (or non-negative) integer.

    Parameters
    ----------
    value : Any
        Value to validate.
    param : str
        Parameter name used in error messages.
    allow_zero : bool, default False
        If True, zero is accepted in addition to positive integers.

    Raises
    ------
    TypeError
        If *value* is not an integer.
    ValueError
        If *value* does not satisfy the positivity constraint.

    Examples
    --------
    >>> from missingly._validation import validate_positive_int
    >>> validate_positive_int(5, param="n_neighbors")    # passes
    >>> validate_positive_int(0, param="max_iter")
    Traceback (most recent call last):
        ...
    ValueError: `max_iter` must be a positive integer (> 0); got 0.
    """
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"`{param}` must be an integer; got {type(value).__name__}."
        )
    threshold = 0 if allow_zero else 1
    comparator = ">= 0" if allow_zero else "> 0"
    qualifier = "non-negative" if allow_zero else "positive"
    if value < threshold:
        raise ValueError(
            f"`{param}` must be a {qualifier} integer ({comparator}); got {value}."
        )
