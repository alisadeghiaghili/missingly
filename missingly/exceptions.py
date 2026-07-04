"""Domain exceptions for the missingly package.

All public exceptions inherit from :class:`MissinglyError` so callers can
catch the entire family with a single ``except MissinglyError`` clause, or
catch specific sub-classes for fine-grained handling.

Hierarchy
---------
::

    MissinglyError
    ├── ImputationError          # estimator fit/predict failed
    ├── InsufficientDataError    # too few observed values to impute
    ├── InvalidStrategyError     # unknown or mis-configured strategy name
    ├── MissingColumnError       # requested column not found in DataFrame
    └── ConfigurationError       # invalid config value or combination

Examples
--------
>>> from missingly.exceptions import InvalidStrategyError
>>> raise InvalidStrategyError(
...     param="strategy",
...     got="avg",
...     allowed=["mean", "median", "knn"],
... )
Traceback (most recent call last):
    ...
missingly.exceptions.InvalidStrategyError: `strategy` must be one of ['mean', 'median', 'knn']; got 'avg'.
"""

from __future__ import annotations

from typing import Any, List, Optional


class MissinglyError(Exception):
    """Base class for all missingly exceptions."""


class ImputationError(MissinglyError):
    """Raised when an estimator fails during fit or predict.

    Parameters
    ----------
    column : str
        Name of the column being imputed when the failure occurred.
    strategy : str
        Imputation strategy that was active.
    original : Exception
        The underlying exception from the estimator.

    Examples
    --------
    >>> err = ImputationError(column="age", strategy="rf", original=ValueError("bad"))
    >>> str(err)
    "Imputation failed for column 'age' using strategy 'rf': bad"
    """

    def __init__(self, column: str, strategy: str, original: Exception) -> None:
        self.column = column
        self.strategy = strategy
        self.original = original
        super().__init__(
            f"Imputation failed for column {column!r} using strategy {strategy!r}: "
            f"{original}"
        )


class InsufficientDataError(MissinglyError):
    """Raised when a column has too few observed values to support imputation.

    Parameters
    ----------
    column : str
        Name of the column with insufficient data.
    n_observed : int
        Number of non-missing values found.
    n_required : int
        Minimum number required by the strategy.

    Examples
    --------
    >>> err = InsufficientDataError(column="salary", n_observed=0, n_required=1)
    >>> str(err)
    "Column 'salary' has 0 observed value(s), but at least 1 is required."
    """

    def __init__(self, column: str, n_observed: int, n_required: int) -> None:
        self.column = column
        self.n_observed = n_observed
        self.n_required = n_required
        super().__init__(
            f"Column {column!r} has {n_observed} observed value(s), "
            f"but at least {n_required} is required."
        )


class InvalidStrategyError(MissinglyError):
    """Raised when an unknown or invalid strategy name is supplied.

    Parameters
    ----------
    param : str
        Name of the parameter that received the bad value.
    got : Any
        The value that was passed.
    allowed : list
        The list of accepted values.
    example : str, optional
        A ready-to-use example value shown in the error message.

    Examples
    --------
    >>> err = InvalidStrategyError(param="strategy", got="avg", allowed=["mean", "median"])
    >>> str(err)
    "`strategy` must be one of ['mean', 'median']; got 'avg'."
    """

    def __init__(
        self,
        param: str,
        got: Any,
        allowed: List[Any],
        example: Optional[str] = None,
    ) -> None:
        self.param = param
        self.got = got
        self.allowed = allowed
        msg = f"`{param}` must be one of {allowed}; got {got!r}."
        if example:
            msg += f"  Try: {param}={example!r}"
        super().__init__(msg)


class MissingColumnError(MissinglyError):
    """Raised when a requested column does not exist in the DataFrame.

    Parameters
    ----------
    columns : list of str
        Column name(s) that were not found.
    available : list of str
        Column names that are present in the DataFrame.

    Examples
    --------
    >>> err = MissingColumnError(columns=["age"], available=["name", "salary"])
    >>> str(err)
    "Column(s) not found in DataFrame: ['age']. Available columns: ['name', 'salary']"
    """

    def __init__(self, columns: List[str], available: List[str]) -> None:
        self.columns = columns
        self.available = available
        super().__init__(
            f"Column(s) not found in DataFrame: {columns}. "
            f"Available columns: {available}"
        )


class ConfigurationError(MissinglyError):
    """Raised when a configuration value is invalid or an invalid combination is used.

    Parameters
    ----------
    param : str
        Name of the config parameter that is invalid.
    got : Any
        The invalid value that was supplied.
    reason : str
        Human-readable explanation of why the value is invalid.

    Examples
    --------
    >>> err = ConfigurationError(param="large_df_threshold", got=-1, reason="must be a positive integer")
    >>> str(err)
    "Invalid configuration for `large_df_threshold` (got -1): must be a positive integer"
    """

    def __init__(self, param: str, got: Any, reason: str) -> None:
        self.param = param
        self.got = got
        self.reason = reason
        super().__init__(
            f"Invalid configuration for `{param}` (got {got!r}): {reason}"
        )
