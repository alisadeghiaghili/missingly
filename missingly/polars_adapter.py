"""Native Polars operations for scalable missingness exploration.

The functions in this module intentionally accept only Polars objects. They do
not silently convert a frame to pandas, because that conversion can eagerly
materialize a large or lazy data set and undermine the caller's memory plan.
"""

from __future__ import annotations

from typing import Any

__all__ = ["polars_missing_summary"]


def _require_polars() -> Any:
    """Import Polars only when a Polars-specific API is called.

    Returns
    -------
    module
        The imported :mod:`polars` module.

    Raises
    ------
    ImportError
        If the optional ``polars`` extra is not installed.

    Examples
    --------
    >>> module = _require_polars()
    >>> module.__name__
    'polars'
    """
    try:
        import polars as pl
    except ImportError as error:
        raise ImportError(
            "polars_missing_summary requires the optional 'polars' extra. "
            "Install it with `pip install missingly[polars]`."
        ) from error
    return pl


def polars_missing_summary(frame: Any) -> Any:
    """Summarize nulls in an eager or lazy Polars frame without pandas copies.

    The function treats only native Polars nulls as missing. It returns one row
    per input column with the number and percentage of null and non-null values.
    When given a :class:`polars.LazyFrame`, the returned LazyFrame retains lazy
    execution; call ``collect()`` explicitly to materialize the small summary.

    Parameters
    ----------
    frame : polars.DataFrame or polars.LazyFrame
        Eager or lazy Polars data to summarize. A pandas DataFrame is rejected
        to prevent an implicit, potentially large in-memory conversion.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        A result of the same execution mode as ``frame`` with columns
        ``variable``, ``n_missing``, ``n_complete``, and ``pct_missing``.

    Raises
    ------
    ImportError
        If Polars is not installed.
    TypeError
        If ``frame`` is not a Polars DataFrame or LazyFrame.

    Examples
    --------
    >>> import polars as pl
    >>> frame = pl.DataFrame({"score": [1.0, None, 3.0]})
    >>> polars_missing_summary(frame).to_dicts()
    [{'variable': 'score', 'n_missing': 1, 'n_complete': 2, 'pct_missing': 33.333333333333336}]
    >>> lazy_result = polars_missing_summary(frame.lazy())
    >>> isinstance(lazy_result, pl.LazyFrame)
    True
    """
    pl = _require_polars()
    if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            "frame must be a polars DataFrame or LazyFrame; "
            f"got {type(frame).__name__}."
        )

    is_lazy = isinstance(frame, pl.LazyFrame)
    lazy_frame = frame if is_lazy else frame.lazy()
    columns = lazy_frame.collect_schema().names()
    if not columns:
        empty = pl.DataFrame(
            schema={
                "variable": pl.String,
                "n_missing": pl.UInt32,
                "n_complete": pl.UInt32,
                "pct_missing": pl.Float64,
            }
        )
        return empty.lazy() if is_lazy else empty

    summaries = [
        lazy_frame.select(
            pl.lit(column).alias("variable"),
            pl.col(column).null_count().alias("n_missing"),
            (pl.len() - pl.col(column).null_count()).alias("n_complete"),
            (pl.col(column).null_count() * 100.0 / pl.len()).alias("pct_missing"),
        )
        for column in columns
    ]
    result = pl.concat(summaries)
    return result if is_lazy else result.collect()
