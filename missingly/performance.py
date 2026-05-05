"""Performance utilities for large-dataset workflows.

This module provides three complementary tools that make missingly
usable on DataFrames that do not fit comfortably in memory or that
take too long to process in a single pass:

:func:`chunk_apply`
    Applies any missingly function (or arbitrary callable) to a
    DataFrame in fixed-size row chunks and concatenates the results.
    Drop-in replacement for a direct call when you need to stay within
    a memory budget.

:func:`memory_usage_mb`
    Returns a tidy summary of per-column and total memory usage in
    megabytes.  Useful for deciding whether chunking is necessary.

:func:`optimize_dtypes`
    Downcasts numeric columns to the smallest dtype that preserves all
    values, and optionally converts low-cardinality string/object
    columns to ``pd.Categorical``.  Can reduce DataFrame memory by
    50–80% before any processing.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def memory_usage_mb(
    df: pd.DataFrame,
    deep: bool = True,
) -> pd.DataFrame:
    """Return per-column and total memory usage in megabytes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    deep : bool, optional
        If True (default), introspect object columns for their true
        memory footprint (more accurate but slightly slower).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``dtype``, ``memory_mb``, indexed by
        column name.  The last row is a synthetic ``__total__`` row
        with the sum across all columns.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': np.arange(1_000_000), 'b': np.random.rand(1_000_000)})
    >>> memory_usage_mb(df)
    """
    usage = df.memory_usage(deep=deep, index=False)
    dtypes = df.dtypes

    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(dtypes[col]),
            "memory_mb": usage[col] / (1024 ** 2),
        })

    result = pd.DataFrame(rows).set_index("column")
    total_mb = result["memory_mb"].sum()
    total_row = pd.DataFrame(
        [{"dtype": "—", "memory_mb": total_mb}],
        index=pd.Index(["__total__"], name="column"),
    )
    return pd.concat([result, total_row])


def optimize_dtypes(
    df: pd.DataFrame,
    categorical_threshold: Optional[float] = 0.50,
    downcast_int: bool = True,
    downcast_float: bool = True,
) -> pd.DataFrame:
    """Downcast numeric dtypes and optionally convert low-cardinality object columns.

    This function reduces DataFrame memory usage without losing any
    information:

    * Integer columns are downcast to the smallest signed int dtype
      (int8 → int16 → int32 → int64) that can represent the column's
      min/max values.
    * Float columns are downcast from float64 to float32 where the
      precision loss is acceptable (i.e. no value changes after
      round-trip through float32).
    * Object/string columns whose cardinality (unique / total) is
      below *categorical_threshold* are converted to ``pd.Categorical``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Returned as a copy; never mutated.
    categorical_threshold : float or None, optional
        Fraction of unique values below which an object column is
        converted to Categorical (default 0.50, i.e. < 50% unique).
        Set to ``None`` to disable categorical conversion.
    downcast_int : bool, optional
        Whether to downcast integer columns.  Default True.
    downcast_float : bool, optional
        Whether to attempt float64 → float32 downcast.  Default True.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with optimised dtypes.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     'age':  np.array([25, 30, 35], dtype='int64'),
    ...     'city': ['Paris', 'Lyon', 'Paris'],
    ... })
    >>> opt = optimize_dtypes(df)
    >>> opt.dtypes
    age      int8
    city     category
    dtype: object
    """
    result = df.copy()

    for col in result.columns:
        col_dtype = result[col].dtype

        if downcast_int and pd.api.types.is_integer_dtype(col_dtype):
            col_min = result[col].min()
            col_max = result[col].max()
            for target in (np.int8, np.int16, np.int32, np.int64):
                info = np.iinfo(target)
                if info.min <= col_min and col_max <= info.max:
                    result[col] = result[col].astype(target)
                    break

        elif downcast_float and col_dtype == np.float64:
            as_f32 = result[col].astype(np.float32)
            # Only downcast if round-trip is lossless for non-null values
            mask = result[col].notna()
            if np.allclose(
                result.loc[mask, col].to_numpy(),
                as_f32[mask].astype(np.float64).to_numpy(),
                equal_nan=True,
            ):
                result[col] = as_f32

        elif (
            categorical_threshold is not None
            and col_dtype == object
            and len(result) > 0
        ):
            cardinality = result[col].nunique(dropna=True) / len(result)
            if cardinality < categorical_threshold:
                result[col] = result[col].astype("category")

    return result


def chunk_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10_000,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Apply a function to a DataFrame in fixed-size row chunks.

    Splits *df* into consecutive row chunks of *chunk_size*, applies
    *func* to each chunk independently, then concatenates the results.
    The primary use case is applying memory-intensive missingly functions
    (e.g. imputation, `miss_var_summary`) to DataFrames too large to
    process in a single pass.

    .. note::
        Functions that compute *global* statistics (e.g. `mcar_test`,
        `compare_imputations`) should **not** be used with
        `chunk_apply`; their results are not composable across chunks.
        Use `chunk_apply` only for row-wise transformations where each
        row’s output depends only on that row’s input (or on per-chunk
        statistics, as in mean imputation — which will give slightly
        different fill values per chunk).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Not mutated.
    func : callable
        Function ``func(chunk: pd.DataFrame) -> pd.DataFrame``.
        Must return a DataFrame with the same columns.
    chunk_size : int, optional
        Number of rows per chunk.  Default 10 000.
    reset_index : bool, optional
        If True (default), reset and drop the index of the concatenated
        result so the output has a clean 0-based RangeIndex.

    Returns
    -------
    pd.DataFrame
        Concatenated result of applying *func* to each chunk.

    Raises
    ------
    ValueError
        If *chunk_size* < 1.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> from missingly.impute import impute_mean
    >>> from missingly.performance import chunk_apply
    >>>
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'a': rng.normal(size=50_000),
    ...                    'b': rng.normal(size=50_000)})
    >>> df.loc[rng.choice(50_000, 5_000, replace=False), 'a'] = np.nan
    >>> df_imputed = chunk_apply(df, impute_mean, chunk_size=10_000)
    >>> df_imputed['a'].isnull().sum()
    0
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size!r}")

    n = len(df)
    if n == 0:
        return func(df)

    chunks = []
    for start in range(0, n, chunk_size):
        chunk = df.iloc[start: start + chunk_size]
        chunks.append(func(chunk))

    result = pd.concat(chunks, ignore_index=reset_index)
    return result
