"""Internal dtype normalisation layer for the imputers in :mod:`missingly.impute`.

pandas has many ways to store the "same" column: ``int64`` and ``Int64``,
``float32`` and ``Float64``, ``bool`` and ``boolean``, ``object`` and
``string`` and ``category``, ``datetime64`` with or without a time zone, and
so on.  Passing such columns straight to scikit-learn or to NumPy either
crashes (``pd.NA`` has no float value), silently changes dtypes (an ``int64``
ID column comes back as ``float64``) or silently changes meaning (timestamps
treated as ordinal categories).

This module is the single place that decides what each column *is* and how it
travels through an imputer:

``classify_column``
    Map a column to a :class:`ColumnKind`.
``normalize_frame``
    Produce the *modelled frame* an imputer works on: numeric columns as
    ``float64`` with ``NaN`` for missing; boolean and categorical columns left
    in their own dtype for the existing ordinal encoder; "opaque" columns
    (datetime, timedelta, mixed-type object) removed and kept aside.
``restore_dtypes``
    Put the imputed result back into the original dtypes: complete columns
    are copied from the input verbatim, imputed numeric columns are cast back
    cell by cell, and opaque columns are re-attached.

Dtype rules (also documented in the public imputers)
----------------------------------------------------
* **Numeric kinds** (``int*``, ``uint*``, ``Int*``, ``UInt*``, ``float*``,
  ``Float*``) are modelled as ``float64``.
* **Integer targets** imputed by a conditional model (KNN, MICE, random
  forest, gradient boosting, PMM) are rounded to the nearest integer
  (``numpy.rint``, ties to even) and clipped to the dtype range, so ``Int64``
  stays ``Int64``.  Mean and median imputation never round: a nullable
  integer column keeps its dtype when every fill value is integral and is
  returned as ``Float64`` otherwise (NumPy integer columns cannot hold a
  missing value in the first place).
* **Boolean and categorical kinds** (``bool``, ``boolean``, ``category``,
  ``string``, ``object``) are imputed as categories and keep their dtype.
* **Opaque kinds** (datetime with or without a time zone, timedelta, and
  ``object`` columns that mix value types) are never modelled.  A complete
  opaque column passes through unchanged and is not used as a predictor; an
  incomplete one raises :class:`~missingly.exceptions.UnsupportedDtypeError`
  from the model-based imputers.  Mode-based imputation supports every kind.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Collection, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd

from missingly.exceptions import UnsupportedDtypeError

__all__ = [
    "ColumnKind",
    "NUMERIC_KINDS",
    "NormalizedFrame",
    "classify_column",
    "classify_frame",
    "column_mode",
    "ensure_supported",
    "fill_missing",
    "fill_with_statistic",
    "is_numeric_kind",
    "normalize_frame",
    "restore_dtypes",
    "restore_numeric",
    "to_float_array",
]

_MIXED_INFERRED_TYPES = frozenset({"mixed", "mixed-integer"})


class ColumnKind(str, enum.Enum):
    """How an imputer treats a column.

    Examples
    --------
    >>> ColumnKind.INTEGER.value
    'integer'
    """

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    MIXED = "mixed"
    OTHER = "other"


NUMERIC_KINDS = frozenset({ColumnKind.INTEGER, ColumnKind.FLOAT})
_OPAQUE_REASONS: Dict[ColumnKind, str] = {
    ColumnKind.DATETIME: "datetime columns are not modelled",
    ColumnKind.TIMEDELTA: "timedelta columns are not modelled",
    ColumnKind.MIXED: "object columns mixing value types are not modelled",
    ColumnKind.OTHER: "this dtype is not modelled",
}


def is_numeric_kind(kind: ColumnKind) -> bool:
    """Return whether *kind* is modelled as a real-valued column.

    Parameters
    ----------
    kind : ColumnKind
        Kind returned by :func:`classify_column`.

    Returns
    -------
    bool
        ``True`` for :attr:`ColumnKind.INTEGER` and :attr:`ColumnKind.FLOAT`.

    Examples
    --------
    >>> is_numeric_kind(ColumnKind.FLOAT), is_numeric_kind(ColumnKind.BOOLEAN)
    (True, False)
    """
    return kind in NUMERIC_KINDS


def classify_column(series: pd.Series) -> ColumnKind:
    """Classify one column by dtype (and by content for ``object`` columns).

    Parameters
    ----------
    series : pandas.Series
        Column to classify.

    Returns
    -------
    ColumnKind
        ``CATEGORICAL`` for ``category``, ``string`` and homogeneous
        ``object`` columns; ``BOOLEAN`` for ``bool`` and ``boolean``;
        ``INTEGER`` / ``FLOAT`` for NumPy and nullable numeric dtypes;
        ``DATETIME`` (including time-zone aware), ``TIMEDELTA``; ``MIXED``
        for ``object`` columns whose non-missing values mix strings and
        numbers; ``OTHER`` for complex numbers.

    Examples
    --------
    >>> classify_column(pd.Series([1, None], dtype="Int64")).value
    'integer'
    >>> classify_column(pd.Series([True, False])).value
    'boolean'
    >>> classify_column(pd.Series([1, "a"], dtype=object)).value
    'mixed'
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _classify_object_like(series)
    checks = (
        (pd.api.types.is_bool_dtype, ColumnKind.BOOLEAN),
        (pd.api.types.is_datetime64_any_dtype, ColumnKind.DATETIME),
        (pd.api.types.is_timedelta64_dtype, ColumnKind.TIMEDELTA),
        (pd.api.types.is_complex_dtype, ColumnKind.OTHER),
        (pd.api.types.is_integer_dtype, ColumnKind.INTEGER),
        (pd.api.types.is_float_dtype, ColumnKind.FLOAT),
    )
    for predicate, kind in checks:
        if predicate(dtype):
            return kind
    return _classify_object_like(series)


def _classify_object_like(series: pd.Series) -> ColumnKind:
    """Return ``MIXED`` for value-type-mixing object data, else ``CATEGORICAL``."""
    if series.dtype != object and not isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL
    inferred = pd.api.types.infer_dtype(series.astype(object), skipna=True)
    if inferred in _MIXED_INFERRED_TYPES:
        return ColumnKind.MIXED
    return ColumnKind.CATEGORICAL


def classify_frame(df: pd.DataFrame) -> Dict[Hashable, ColumnKind]:
    """Classify every column of *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to inspect.

    Returns
    -------
    dict
        Column label to :class:`ColumnKind`, in column order.

    Examples
    --------
    >>> kinds = classify_frame(pd.DataFrame({"a": [1.5], "b": ["x"]}))
    >>> [kind.value for kind in kinds.values()]
    ['float', 'categorical']
    """
    return {column: classify_column(series) for column, series in df.items()}


def to_float_array(series: pd.Series) -> np.ndarray:
    """Return a numeric column as ``float64`` with ``NaN`` for missing cells.

    Parameters
    ----------
    series : pandas.Series
        NumPy or nullable numeric column.

    Returns
    -------
    numpy.ndarray
        ``float64`` array; ``pd.NA`` becomes ``numpy.nan``.

    Examples
    --------
    >>> to_float_array(pd.Series([1, None], dtype="Int64")).tolist()
    [1.0, nan]
    """
    return series.to_numpy(dtype="float64", na_value=np.nan, copy=True)


def column_mode(series: pd.Series, kind: Optional[ColumnKind] = None) -> Any:
    """Return the most frequent observed value of a column.

    Parameters
    ----------
    series : pandas.Series
        Column with at least one observed value.
    kind : ColumnKind, optional
        Pre-computed kind.  Columns of kind ``MIXED`` cannot be sorted, so
        ties are broken by first occurrence; every other kind breaks ties by
        the smallest value (``Series.mode`` order).

    Returns
    -------
    object
        The modal value.

    Examples
    --------
    >>> column_mode(pd.Series(["b", "a", "b", None]))
    'b'
    >>> column_mode(pd.Series([2, "a", "a", 2], dtype=object))
    2
    """
    observed = series.dropna()
    if kind is None:
        kind = classify_column(series)
    if kind is ColumnKind.MIXED:
        return observed.value_counts(sort=False).idxmax()
    return observed.mode().iloc[0]


def fill_missing(series: pd.Series, value: Any) -> pd.Series:
    """Fill the missing cells of one column with a scalar, keeping its dtype.

    Parameters
    ----------
    series : pandas.Series
        Column to fill; not modified.
    value : object
        Scalar already valid for the column dtype, typically a mode.

    Returns
    -------
    pandas.Series
        Filled copy; observed cells are unchanged.

    Examples
    --------
    >>> fill_missing(pd.Series([1, None], dtype="Int64"), 1).dtype
    Int64Dtype()
    >>> fill_missing(pd.Series(["a", None]), "a").tolist()
    ['a', 'a']
    """
    missing = series.isna()
    if not missing.any():
        return series.copy()
    return series.where(~missing, value)


def fill_with_statistic(series: pd.Series, value: float) -> pd.Series:
    """Fill a numeric column with a real-valued statistic such as a mean.

    The fill goes through ``float64`` and is cast back without rounding: a
    nullable integer column keeps its dtype when the statistic is integral
    and becomes ``Float64`` otherwise (instead of raising).  Columns of any
    other kind fall back to :func:`fill_missing`.

    Parameters
    ----------
    series : pandas.Series
        Column to fill; not modified.
    value : float
        Real-valued fill.

    Returns
    -------
    pandas.Series
        Filled copy; observed cells are unchanged.

    Examples
    --------
    >>> fill_with_statistic(pd.Series([1, None], dtype="Int64"), 1.5).tolist()
    [1.0, 1.5]
    >>> fill_with_statistic(pd.Series([1, None], dtype="Int64"), 2.0).dtype
    Int64Dtype()
    """
    missing = series.isna()
    kind = classify_column(series)
    if not missing.any() or not is_numeric_kind(kind):
        return fill_missing(series, value)
    filled = to_float_array(series)
    filled[missing.to_numpy()] = float(value)
    return restore_numeric(
        series, pd.Series(filled, index=series.index), kind, round_integers=False
    )


def _clip_to_integer_range(values: np.ndarray, dtype: Any) -> np.ndarray:
    """Clip *values* to the representable range of integer *dtype*."""
    info = np.iinfo(getattr(dtype, "numpy_dtype", dtype))
    lower = float(info.min)
    upper = float(info.max)
    if int(upper) > info.max:  # 64-bit maxima round up to 2**63 / 2**64
        upper = float(np.nextafter(upper, 0.0))
    return np.clip(values, lower, upper)


def restore_numeric(
    original: pd.Series,
    imputed: pd.Series,
    kind: ColumnKind,
    *,
    round_integers: bool = True,
) -> pd.Series:
    """Write the imputed cells of a numeric column back into its original dtype.

    Only cells that were missing in *original* are taken from *imputed*; every
    observed cell keeps its exact original value, even for 64-bit integers
    that ``float64`` cannot represent.

    Parameters
    ----------
    original : pandas.Series
        Column before imputation (any numeric dtype, possibly with missing
        cells).
    imputed : pandas.Series
        Same-length column of imputed values (``float64`` or nullable).
    kind : ColumnKind
        Kind of *original*.
    round_integers : bool, default True
        Policy for integer kinds.  ``True``: round imputed cells to the
        nearest integer (ties to even), clip them to the dtype range and keep
        the original dtype.  ``False``: keep the integer dtype only if every
        imputed value is already integral; otherwise return ``Float64`` so
        that no value is altered.  (A NumPy integer column has no missing
        cell, so this policy only ever applies to nullable integers.)

    Returns
    -------
    pandas.Series
        Column with the original index and name.

    Examples
    --------
    >>> original = pd.Series([1, None, 4], dtype="Int64")
    >>> imputed = pd.Series([1.0, 2.6, 4.0])
    >>> restore_numeric(original, imputed, ColumnKind.INTEGER).tolist()
    [1, 3, 4]
    >>> restore_numeric(original, imputed, ColumnKind.INTEGER, round_integers=False).dtype
    Float64Dtype()
    >>> whole = pd.Series([1.0, 3.0, 4.0])
    >>> restore_numeric(original, whole, ColumnKind.INTEGER, round_integers=False).dtype
    Int64Dtype()
    """
    target = original.dtype
    values = to_float_array(imputed)
    if kind is ColumnKind.INTEGER:
        if round_integers:
            values = _clip_to_integer_range(np.rint(values), target)
        elif not _is_integral(values[original.isna().to_numpy()]):
            target = pd.Float64Dtype()
    complete = pd.Series(values, index=original.index, name=original.name).astype(
        target
    )
    return original.astype(target).where(~original.isna(), complete)


def _is_integral(values: np.ndarray) -> bool:
    """Return whether every non-NaN entry of *values* is a whole number."""
    finite = values[~np.isnan(values)]
    return bool(np.all(finite == np.rint(finite)))


@dataclass(frozen=True)
class NormalizedFrame:
    """A frame split into the part an imputer models and the part it must not touch.

    Attributes
    ----------
    frame : pandas.DataFrame
        Modelled columns in input order.  Numeric columns are ``float64`` with
        ``NaN`` for missing; boolean and categorical columns keep their dtype.
    original : pandas.DataFrame
        The input frame, used to restore complete and opaque columns.
    kinds : dict
        Kind of every input column.
    passthrough : tuple
        Labels of opaque columns that are not part of ``frame``.
    """

    frame: pd.DataFrame
    original: pd.DataFrame
    kinds: Dict[Hashable, ColumnKind]
    passthrough: Tuple[Hashable, ...]


def ensure_supported(
    df: pd.DataFrame,
    strategy: str,
    targets: Optional[Collection[Hashable]] = None,
    kinds: Optional[Dict[Hashable, ColumnKind]] = None,
) -> Dict[Hashable, ColumnKind]:
    """Reject opaque columns that a model-based imputer would have to impute.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (Python ``None`` already normalised to ``NaN``).
    strategy : str
        Imputer name used in the error message, for example ``"knn"``.
    targets : collection of str, optional
        Columns the caller will impute.  Defaults to every column with a
        missing value.
    kinds : dict, optional
        Pre-computed result of :func:`classify_frame`.

    Returns
    -------
    dict
        Kind of every column.

    Raises
    ------
    UnsupportedDtypeError
        If a datetime, timedelta or mixed-type object target column has
        missing values.

    Examples
    --------
    >>> frame = pd.DataFrame({"t": pd.to_timedelta([1, None], unit="D")})
    >>> try:
    ...     ensure_supported(frame, "knn")
    ... except UnsupportedDtypeError as error:
    ...     print(error.column, error.strategy)
    t knn
    """
    kinds = classify_frame(df) if kinds is None else kinds
    wanted = set(df.columns[df.isna().any(axis=0)]) if targets is None else set(targets)
    for column, kind in kinds.items():
        if kind in _OPAQUE_REASONS and column in wanted and df[column].isna().any():
            raise UnsupportedDtypeError(
                column=str(column),
                dtype=df[column].dtype,
                strategy=strategy,
                reason=_OPAQUE_REASONS[kind],
            )
    return kinds


def normalize_frame(
    df: pd.DataFrame,
    strategy: str,
    targets: Optional[Collection[Hashable]] = None,
) -> NormalizedFrame:
    """Prepare *df* for a model-based imputer.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (Python ``None`` already normalised to ``NaN``).
    strategy : str
        Imputer name used in error messages, for example ``"knn"``.
    targets : collection of str, optional
        Columns the caller will actually impute.  Defaults to every column
        with a missing value.  An opaque column with missing values outside
        *targets* is passed through with its missing cells intact.

    Returns
    -------
    NormalizedFrame
        Modelled frame plus the information needed by :func:`restore_dtypes`.

    Raises
    ------
    UnsupportedDtypeError
        If an opaque (datetime, timedelta, mixed-type object) target column
        has missing values.

    Examples
    --------
    >>> frame = pd.DataFrame({"n": pd.array([1, None], dtype="Int64"), "t": pd.to_timedelta([1, 2], unit="D")})
    >>> normalized = normalize_frame(frame, "knn")
    >>> list(normalized.frame.columns), normalized.passthrough
    (['n'], ('t',))
    >>> str(normalized.frame["n"].dtype)
    'float64'
    """
    kinds = ensure_supported(df, strategy, targets)
    modelled: Dict[Hashable, Any] = {}
    passthrough: List[Hashable] = []
    for column, kind in kinds.items():
        if kind in _OPAQUE_REASONS:
            passthrough.append(column)
        elif is_numeric_kind(kind):
            modelled[column] = to_float_array(df[column])
        else:
            modelled[column] = df[column]
    frame = pd.DataFrame(modelled, index=df.index, columns=list(modelled))
    return NormalizedFrame(frame, df, kinds, tuple(passthrough))


def restore_dtypes(
    imputed: pd.DataFrame,
    normalized: NormalizedFrame,
    *,
    round_integers: bool = True,
) -> pd.DataFrame:
    """Return the imputed frame in the original column dtypes.

    Parameters
    ----------
    imputed : pandas.DataFrame
        Result of imputing ``normalized.frame``; categorical columns must
        already be decoded to their original dtype.
    normalized : NormalizedFrame
        Value returned by :func:`normalize_frame` for the same input.
    round_integers : bool, default True
        Passed to :func:`restore_numeric` for integer columns.

    Returns
    -------
    pandas.DataFrame
        Columns of *imputed* in their existing order, followed by the opaque
        pass-through columns.  Columns that had no missing value are copied
        from the input, so their dtype and values are exactly preserved.

    Examples
    --------
    >>> source = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64"), "k": [1, 2, 3]})
    >>> prepared = normalize_frame(source, "knn")
    >>> filled = prepared.frame.fillna(2.2)
    >>> restored = restore_dtypes(filled, prepared)
    >>> restored.dtypes.astype(str).tolist(), restored["n"].tolist()
    (['Int64', 'int64'], [1, 2, 3])
    """
    original = normalized.original
    columns: Dict[Hashable, pd.Series] = {}
    for column in imputed.columns:
        source = original[column]
        if not source.isna().any():
            columns[column] = source
        elif is_numeric_kind(normalized.kinds[column]):
            columns[column] = restore_numeric(
                source,
                imputed[column],
                normalized.kinds[column],
                round_integers=round_integers,
            )
        else:
            columns[column] = imputed[column]
    for label in normalized.passthrough:
        columns[label] = original[label]
    return pd.DataFrame(columns, index=original.index)
