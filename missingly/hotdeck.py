"""Hot-deck imputation for missingly.

Hot-deck imputation replaces each missing value with an observed value
(a *donor*) drawn from the same dataset.  Unlike mean or regression
imputation it:

* Preserves the empirical marginal distribution of each variable.
* Never produces impossible values (e.g. negative ages, out-of-range codes).
* Works naturally with categorical, ordinal, and numeric columns.

This module provides three variants that cover the most common use-cases
in survey research, epidemiology, and official statistics:

``impute_hotdeck_random``
    Classic *simple random hot-deck*: for each missing cell, sample one
    donor uniformly at random from all rows that have an observed value
    in that column (optionally within imputation classes, ``by=``).
    This is the random hot-deck of Andridge & Little (2010); it matches
    the SRS-style donor draw of SAS ``PROC SURVEYIMPUTE`` and SPSS
    ``MVA`` without their survey-design extras.

``impute_hotdeck_sequential``
    *Sequential (carry-forward / carry-backward) hot-deck*: scan the
    column in order and substitute the closest preceding observed value
    (``direction='forward'``), the closest following observed value
    (``direction='backward'``), or the observed value closest in
    position (``direction='closest'``).  Imputation classes (``by=``)
    keep carried values inside their group, for example one panel
    subject.  Common in time-ordered survey data and panel datasets.

``impute_hotdeck_weighted``
    *Distance-weighted hot-deck*: for each missing row, compute Gower
    distances to all donor rows using all *other* columns as matching
    variables, keep the ``n_donors`` nearest donors (equally near donors
    are ordered at random) and sample one of them with probability
    proportional to ``1 / (1 + distance)``.  This is a stochastic
    nearest-neighbour hot-deck in the spirit of the distance-based donor
    selection reviewed by Andridge & Little (2010); it is not a port of
    VIM's ``hotdeck()`` or of SAS fractional imputation.
    The distance matrix is built with NumPy in chunks, so the cost is
    O(n_missing × n_donors × n_features) per column; a ``UserWarning`` is
    emitted when the DataFrame exceeds ``config.large_df_threshold`` rows
    or when the number of recipient-donor pairs becomes very large.

References
----------
Kalton, G. & Kasprzyk, D. (1986). The treatment of missing survey data.
    *Survey Methodology*, 12(1), 1-16.
Andridge, R.R. & Little, R.J.A. (2010). A review of hot deck imputation
    for survey non-response. *International Statistical Review*, 78(1),
    40-64. https://doi.org/10.1111/j.1751-5823.2010.00103.x
Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.),
    Ch. 3.4. https://stefvanbuuren.name/fimd/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Sequence, Union

import numpy as np
import pandas as pd

from missingly._validation import (
    as_generator,
    validate_dataframe,
    validate_positive_int,
)
from missingly.config import config as _config
from missingly.impute import _normalize_missing, _restore_categorical_dtype


__all__ = [
    "impute_hotdeck_random",
    "impute_hotdeck_sequential",
    "impute_hotdeck_weighted",
]

#: Recipient-donor pairs above which ``impute_hotdeck_weighted`` warns.
_PAIR_WARN_THRESHOLD = 25_000_000
#: Upper bound on the number of cells of one distance block held in memory.
_CHUNK_ELEMENTS = 2_000_000
#: Directions accepted by ``impute_hotdeck_sequential``.
_DIRECTIONS = ("forward", "backward", "nearest", "forward_then_backward", "closest")

_NO_DONOR_MESSAGE = (
    "Column {col!r} has no observed values{where}; cannot hot-deck impute."
)

ByType = Union[str, Sequence[str], None]


# ---------------------------------------------------------------------------
# Imputation classes
# ---------------------------------------------------------------------------


def _normalize_by(df: pd.DataFrame, by: ByType) -> List[Any]:
    """Validate the ``by`` argument and return it as a list of column labels.

    Parameters
    ----------
    df : pd.DataFrame
        Frame the classes are built from.
    by : str, sequence of str or None
        Class-defining column(s).

    Returns
    -------
    list
        Column labels; empty when *by* is ``None``.

    Raises
    ------
    TypeError
        If *by* is neither a string nor a list or tuple of labels.
    ValueError
        If a label is not a column of *df*.
    """
    if by is None:
        return []
    if isinstance(by, str):
        labels: List[Any] = [by]
    elif isinstance(by, (list, tuple)):
        labels = list(by)
    else:
        raise TypeError(
            f"by must be a column name, a list of column names or None; got {by!r}"
        )
    unknown = [label for label in labels if label not in df.columns]
    if unknown:
        raise ValueError(f"by column(s) {unknown!r} not found in the DataFrame")
    return labels


def _class_rows(df: pd.DataFrame, by: List[Any]) -> List[np.ndarray]:
    """Return the positional row indices of every imputation class.

    Rows whose class key is missing form a class of their own.  Rows keep
    their original order inside each class.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    by : list
        Class-defining column labels; empty means one class with every row.

    Returns
    -------
    list of np.ndarray
        One array of row positions per class.
    """
    if not by:
        return [np.arange(len(df))]
    codes = df.groupby(by, dropna=False, sort=False).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes))[:-1]
    return np.split(order, boundaries)


def _no_donor_error(
    df: pd.DataFrame, col: Any, by: List[Any], rows: np.ndarray
) -> ValueError:
    """Build the error raised when a class has no observed donor.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col : Any
        Column being imputed.
    by : list
        Class-defining column labels.
    rows : np.ndarray
        Row positions of the class without donors.

    Returns
    -------
    ValueError
        Error naming the column and, when classes are used, the class.
    """
    where = ""
    if by:
        key = tuple(df.iloc[int(rows[0])][by])
        where = f" in the class {dict(zip(by, key))!r}"
    return ValueError(_NO_DONOR_MESSAGE.format(col=col, where=where))


def _draw_sources(
    df: pd.DataFrame,
    col: Any,
    by: List[Any],
    classes: List[np.ndarray],
    draw: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Choose a donor row for every missing cell of one column.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame (missing values normalised).
    col : Any
        Column being imputed.
    by : list
        Class-defining column labels.
    classes : list of np.ndarray
        Row positions of every imputation class.
    draw : callable
        ``draw(recipients, donors)`` returns one donor row per recipient;
        called once per class that has both recipients and donors.

    Returns
    -------
    np.ndarray of int, shape (n_rows,)
        Donor row for each missing cell, -1 elsewhere.

    Raises
    ------
    ValueError
        If a class has a missing value but no observed donor.
    """
    missing = df[col].isna().to_numpy()
    source = np.full(len(df), -1, dtype=np.intp)
    for rows in classes:
        recipients = rows[missing[rows]]
        if len(recipients) == 0:
            continue
        donors = rows[~missing[rows]]
        if len(donors) == 0:
            raise _no_donor_error(df, col, by, rows)
        source[recipients] = draw(recipients, donors)
    return source


# ---------------------------------------------------------------------------
# Gower distance (vectorised)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GowerFeature:
    """One matching variable prepared for vectorised Gower distances.

    Attributes
    ----------
    values : np.ndarray
        Floats for numeric features (datetime-like values as nanoseconds),
        integer category codes otherwise.
    observed : np.ndarray of bool
        True where the value is not missing.
    numeric : bool
        Whether the feature is compared by range-scaled absolute difference.
    scale : float
        Observed range of a numeric feature; 0.0 contributes nothing.
    """

    values: np.ndarray
    observed: np.ndarray
    numeric: bool
    scale: float


def _is_gower_numeric(series: pd.Series) -> bool:
    """Return whether Gower compares *series* as a real-valued variable.

    Bool and nullable ``boolean`` columns are matched as categories;
    datetime and timedelta columns are numeric (their position on the time
    axis matters).

    Parameters
    ----------
    series : pd.Series
        Column to classify.

    Returns
    -------
    bool
        True for numeric, datetime-like and timedelta columns that are not
        boolean.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    return bool(
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Return a numeric column as float64 with NaN for missing values.

    Datetimes become nanoseconds since the epoch (time-zone aware values are
    converted to UTC first) and timedeltas become nanoseconds.

    Parameters
    ----------
    series : pd.Series
        Numeric, datetime or timedelta column.

    Returns
    -------
    np.ndarray of float64, shape (n_rows,)
        Values on a common numeric scale.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dtype, "tz", None) is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        stamps = series.to_numpy(dtype="datetime64[ns]")
        return np.where(
            np.isnat(stamps), np.nan, stamps.astype("int64").astype(np.float64)
        )
    if pd.api.types.is_timedelta64_dtype(series):
        spans = series.to_numpy(dtype="timedelta64[ns]")
        return np.where(
            np.isnat(spans), np.nan, spans.astype("int64").astype(np.float64)
        )
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.to_numpy(dtype=np.float64, na_value=np.nan)


def _build_feature(series: pd.Series) -> _GowerFeature:
    """Prepare one column for vectorised Gower matching.

    Parameters
    ----------
    series : pd.Series
        Column of the (missing-normalised) input frame.

    Returns
    -------
    _GowerFeature
        Numeric floats or category codes, the observed mask and the range.
    """
    if _is_gower_numeric(series):
        values = _numeric_values(series)
        observed = ~np.isnan(values)
        scale = (
            float(values[observed].max() - values[observed].min())
            if observed.any()
            else 0.0
        )
        return _GowerFeature(values, observed, True, scale)
    codes, _ = pd.factorize(series)
    return _GowerFeature(codes, codes >= 0, False, 0.0)


def _matching_features(
    features: Sequence[_GowerFeature],
    columns: pd.Index,
    target_idx: int,
    by: List[Any],
) -> List[_GowerFeature]:
    """Select the matching variables of one target column.

    Parameters
    ----------
    features : sequence of _GowerFeature
        One prepared feature per column of the frame.
    columns : pd.Index
        Column labels of the frame.
    target_idx : int
        Position of the column being imputed.
    by : list
        Class-defining column labels.

    Returns
    -------
    list of _GowerFeature
        Features of all columns except the target and the ``by`` columns.
    """
    return [
        feature
        for j, feature in enumerate(features)
        if j != target_idx and columns[j] not in by
    ]


def _gower_distance_block(
    features: Sequence[_GowerFeature],
    recipients: np.ndarray,
    donors: np.ndarray,
) -> np.ndarray:
    """Return Gower distances from recipient rows to donor rows.

    A matcher counts for a recipient-donor pair only when it is observed in
    both rows, and the distance is the mean over the matchers that count;
    a pair with no usable matcher has distance 1.0.

    Parameters
    ----------
    features : sequence of _GowerFeature
        Matching variables (the column being imputed is not among them).
    recipients : np.ndarray of int
        Row positions of the recipients.
    donors : np.ndarray of int
        Row positions of the donors.

    Returns
    -------
    np.ndarray of shape (n_recipients, n_donors)
        Gower distance in [0, 1] for every pair.
    """
    numerator = np.zeros((len(recipients), len(donors)), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for feature in features:
        usable = (
            feature.observed[recipients][:, None] & feature.observed[donors][None, :]
        )
        denominator += usable
        if feature.numeric and feature.scale <= 0.0:
            continue
        left = feature.values[recipients][:, None]
        right = feature.values[donors][None, :]
        if feature.numeric:
            gap = np.abs(left - right) / feature.scale
        else:
            gap = (left != right).astype(np.float64)
        numerator += np.where(usable, gap, 0.0)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0.0,
    )


def _pick_weighted_positions(
    dists: np.ndarray,
    n_donors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one donor per recipient from its *n_donors* nearest candidates.

    The pool is the *n_donors* smallest distances, with equally near donors
    ordered at random; the draw inside the pool has probability
    proportional to ``1 / (1 + distance)``.

    Parameters
    ----------
    dists : np.ndarray of shape (n_recipients, n_candidates)
        Gower distance from every recipient to every candidate donor.
    n_donors : int
        Pool size.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of int, shape (n_recipients,)
        Column of *dists* chosen for every recipient.
    """
    n_candidates = dists.shape[1]
    pool_size = min(n_donors, n_candidates)
    if pool_size < n_candidates:
        tie_breaker = rng.random(dists.shape)
        pool = np.lexsort((tie_breaker, dists), axis=-1)[:, :pool_size]
    else:
        pool = np.broadcast_to(np.arange(n_candidates), dists.shape)
    pool_dists = np.take_along_axis(dists, pool, axis=1)
    cumulative = np.cumsum(1.0 / (1.0 + pool_dists), axis=1)
    threshold = rng.random(len(dists)) * cumulative[:, -1]
    slot = np.minimum((cumulative <= threshold[:, None]).sum(axis=1), pool_size - 1)
    return pool[np.arange(len(dists)), slot]


def _weighted_donor_rows(
    features: Sequence[_GowerFeature],
    recipients: np.ndarray,
    donors: np.ndarray,
    n_donors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick one donor row for every recipient row of one class and column.

    Parameters
    ----------
    features : sequence of _GowerFeature
        Matching variables.
    recipients : np.ndarray of int
        Rows whose value in the target column is missing.
    donors : np.ndarray of int
        Rows whose value in the target column is observed.
    n_donors : int
        Candidate pool size.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of int
        Donor row (position in the frame) for each recipient, in order.
    """
    chunk = max(1, _CHUNK_ELEMENTS // len(donors))
    chosen = np.empty(len(recipients), dtype=np.intp)
    for start in range(0, len(recipients), chunk):
        block = recipients[start : start + chunk]
        dists = _gower_distance_block(features, block, donors)
        chosen[start : start + chunk] = donors[
            _pick_weighted_positions(dists, n_donors, rng)
        ]
    return chosen


def _warn_if_large(n_rows: int, n_pairs: int) -> None:
    """Warn when the weighted hot-deck is expected to be slow.

    Parameters
    ----------
    n_rows : int
        Number of rows of the input frame.
    n_pairs : int
        Largest number of recipient-donor pairs of any column.
    """
    if n_rows > _config.large_df_threshold:
        warnings.warn(
            f"impute_hotdeck_weighted: DataFrame has {n_rows:,} rows. "
            "Weighted hot-deck costs O(n_missing × n_donors) per column and may "
            "be slow. Consider impute_hotdeck_random() for large datasets.",
            UserWarning,
            stacklevel=3,
        )
    elif n_pairs > _PAIR_WARN_THRESHOLD:
        warnings.warn(
            f"impute_hotdeck_weighted: {n_pairs:,} recipient-donor pairs in the "
            "largest column. Weighted hot-deck may be slow. Consider "
            "impute_hotdeck_random() or imputation classes (by=).",
            UserWarning,
            stacklevel=3,
        )


def _pair_count(df: pd.DataFrame, classes: List[np.ndarray]) -> int:
    """Return the largest number of recipient-donor pairs of any column.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    classes : list of np.ndarray
        Row positions of every imputation class.

    Returns
    -------
    int
        Maximum over columns of the summed ``n_missing × n_donors`` per class.
    """
    observed = df.notna().to_numpy()
    worst = 0
    for j in range(observed.shape[1]):
        pairs = 0
        for rows in classes:
            n_observed = int(observed[rows, j].sum())
            pairs += (len(rows) - n_observed) * n_observed
        worst = max(worst, pairs)
    return worst


# ---------------------------------------------------------------------------
# Sequential donor positions
# ---------------------------------------------------------------------------


def _sequential_sources(observed: np.ndarray, direction: str) -> np.ndarray:
    """Return the donor position of every row of one ordered sequence.

    Parameters
    ----------
    observed : np.ndarray of bool
        True where the value is observed.
    direction : str
        ``"forward"``, ``"backward"``, ``"forward_then_backward"`` or
        ``"closest"``.

    Returns
    -------
    np.ndarray of int
        Position of the donor for every row (the row itself when it is
        observed); -1 where no donor exists in the requested direction.
    """
    n = len(observed)
    positions = np.arange(n)
    previous = np.maximum.accumulate(np.where(observed, positions, -1))
    following = np.minimum.accumulate(np.where(observed, positions, n)[::-1])[::-1]
    following = np.where(following == n, -1, following)
    if direction == "forward":
        return previous
    if direction == "backward":
        return following
    if direction == "forward_then_backward":
        return np.where(previous >= 0, previous, following)
    both = (previous >= 0) & (following >= 0)
    prefer_previous = positions - previous <= following - positions
    return np.where(
        both,
        np.where(prefer_previous, previous, following),
        np.where(previous >= 0, previous, following),
    )


def _sequence_donors(
    observed: np.ndarray,
    classes: List[np.ndarray],
    direction: str,
) -> np.ndarray:
    """Return donor positions of one column, computed inside each class.

    Parameters
    ----------
    observed : np.ndarray of bool
        True where the column is observed.
    classes : list of np.ndarray
        Row positions of every imputation class.
    direction : str
        Direction passed to :func:`_sequential_sources`.

    Returns
    -------
    np.ndarray of int
        Frame position of the donor of every row, -1 where there is none.
    """
    donors = np.full(len(observed), -1, dtype=np.intp)
    for rows in classes:
        local = _sequential_sources(observed[rows], direction)
        donors[rows] = np.where(local >= 0, rows[np.maximum(local, 0)], -1)
    return donors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def impute_hotdeck_random(
    df: pd.DataFrame,
    random_state: int | None = None,
    by: ByType = None,
) -> pd.DataFrame:
    """Impute missing values via simple random hot-deck.

    For each column that contains missing values, a donor is drawn
    uniformly at random (with replacement) from the rows that have an
    observed value in that column.  With ``by`` the donor is drawn from
    the recipient's own imputation class.  The procedure is
    column-independent: each column is imputed separately.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    random_state : int, numpy integer, Generator, RandomState or None, default None
        Seed for the random number generator.  Pass an integer for
        reproducible results.
        Accepts ``None``, a non-negative integer of any size (Python or numpy),
        a ``numpy.random.Generator`` or a ``numpy.random.RandomState``; any other
        value raises :class:`~missingly.exceptions.InvalidSeedError`.  A boolean
        is deprecated: it gives a :class:`FutureWarning` and acts as ``1`` or ``0``.
    by : str, list of str or None, default None
        Column(s) that define the imputation classes.  Donors are drawn
        only from rows with the same values in these columns; rows with a
        missing class key form a class of their own.  The ``by`` columns
        are never imputed.  ``None`` draws from the whole column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by randomly sampled
        observed values from the same column (and class).

    Raises
    ------
    InvalidSeedError
        If ``random_state`` is not an accepted seed type.
    TypeError
        If *df* is not a :class:`pandas.DataFrame`, or *by* is not a column
        name, a list of names or ``None``.
    ValueError
        If *df* is empty.
    ValueError
        If any column is entirely missing (no donor available), or a class
        has a missing value but no observed donor, or a ``by`` label is not a
        column.

    Warns
    -----
    FutureWarning
        If ``random_state`` is a boolean; it is treated as ``1`` or ``0``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"age": [25.0, np.nan, 35.0, np.nan]})
    >>> impute_hotdeck_random(df, random_state=0)
        age
    0  25.0
    1  35.0
    2  35.0
    3  35.0
    >>> panel = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, np.nan, 9.0, np.nan]})
    >>> impute_hotdeck_random(panel, random_state=0, by="g")
       g    v
    0  a  1.0
    1  a  1.0
    2  b  9.0
    3  b  9.0

    Notes
    -----
    Donor-selection rule: for a missing cell in column *c* (and class *k*),
    the donor is one of the ``m`` rows of the class that have *c* observed,
    each chosen with probability ``1 / m``, independently for every missing
    cell (sampling with replacement).  Imputed values are therefore always
    observed values of the same column and class.

    Because donors are sampled with replacement the marginal distribution of
    each column is preserved in expectation, but variance may be
    underestimated for small donor pools.  For variance-correct inference,
    use multiple imputation (``impute_mice`` with ``n_imputations > 1``)
    instead.  The rule is the random hot-deck described by Andridge and
    Little (2010); it does not use survey weights.

    Donors are copied as observed values, so every column keeps its dtype:
    bool, nullable ``boolean``, ``Int64`` and time-zone aware datetimes are
    returned unchanged in type.

    References
    ----------
    Andridge, R.R. & Little, R.J.A. (2010). A review of hot deck
    imputation for survey non-response. *International Statistical
    Review*, 78(1), 40-64.
    https://doi.org/10.1111/j.1751-5823.2010.00103.x
    """
    validate_dataframe(df, param="df")
    by_cols = _normalize_by(df, by)
    df = _normalize_missing(df)
    rng = as_generator(random_state)
    result = df.copy()
    classes = _class_rows(df, by_cols)

    for col in df.columns:
        if col in by_cols:
            continue
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue
        missing = missing_mask.to_numpy()

        def draw(recipients: np.ndarray, donors: np.ndarray) -> np.ndarray:
            picks = rng.choice(len(donors), size=len(recipients), replace=True)
            return donors[picks]

        source = _draw_sources(df, col, by_cols, classes, draw)
        result.loc[missing_mask, col] = df[col].iloc[source[missing]].to_numpy()
        result[col] = _restore_categorical_dtype(result[col], df[col].dtype)

    return result


def impute_hotdeck_sequential(
    df: pd.DataFrame,
    direction: Literal[
        "forward", "backward", "nearest", "forward_then_backward", "closest"
    ] = "nearest",
    by: ByType = None,
) -> pd.DataFrame:
    """Impute missing values via sequential (carry-forward/backward) hot-deck.

    Scans each column in row order and fills each missing cell with an
    observed value found in the specified direction.  With ``by`` the scan
    runs separately inside each class, so a carried value never crosses from
    one class (for example one panel subject) to the next.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, in the order in which values are carried.
    direction : {"forward", "backward", "nearest", "forward_then_backward", "closest"}, default ``"nearest"``
        Direction in which to search for a donor:

        * ``"forward"``  — use the last seen observed value (LOCF).
          Leaves leading NaN unfilled.
        * ``"backward"`` — use the next observed value (NOCB).
          Leaves trailing NaN unfilled.
        * ``"forward_then_backward"`` — forward first, then backward for
          the leading NaN, so that no NaN remains as long as the column has
          at least one observed value.
        * ``"closest"`` — the observed value nearest in row position;
          when the previous and the next observed value are equally far,
          the previous one is used.
        * ``"nearest"`` — currently identical to
          ``"forward_then_backward"`` and kept for backward compatibility.
          A :class:`FutureWarning` is emitted when its result differs from
          ``"closest"``; pass ``"forward_then_backward"`` or ``"closest"``
          explicitly.
    by : str, list of str or None, default None
        Column(s) that define the classes inside which values are carried.
        Rows with a missing class key form a class of their own.  The ``by``
        columns are never imputed.  Rows are processed in their frame order.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values filled by carry-forward /
        carry-backward substitution.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`, or *by* is not a column
        name, a list of names or ``None``.
    ValueError
        If *df* is empty.
    ValueError
        If *direction* is not one of the supported values, or a ``by`` label
        is not a column.

    Warns
    -----
    FutureWarning
        If ``direction="nearest"`` fills an interior gap differently from
        ``direction="closest"``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"score": [np.nan, 10.0, np.nan, 30.0]})
    >>> impute_hotdeck_sequential(df, direction="forward")
       score
    0    NaN
    1   10.0
    2   10.0
    3   30.0
    >>> impute_hotdeck_sequential(df, direction="closest")
       score
    0   10.0
    1   10.0
    2   10.0
    3   30.0
    >>> panel = pd.DataFrame({"id": [1, 1, 2, 2], "v": [5.0, np.nan, np.nan, 7.0]})
    >>> impute_hotdeck_sequential(panel, direction="forward", by="id")
       id    v
    0   1  5.0
    1   1  5.0
    2   2  NaN
    3   2  7.0

    Notes
    -----
    Donor-selection rule: the donor of a missing cell is the observed value
    at the nearest earlier row (``forward``), the nearest later row
    (``backward``), or the nearest row in either direction (``closest``),
    always inside the same class.  No randomness is involved.

    Forward fill (LOCF) is equivalent to pandas ``ffill()`` and is
    common in time-ordered panel datasets where the last observed
    value is the best proxy for the next period.  It assumes data are
    MAR or MCAR; using LOCF under MNAR inflates bias.  On long-format panel
    data pass the subject identifier as ``by``.

    References
    ----------
    Andridge, R.R. & Little, R.J.A. (2010). A review of hot deck
    imputation for survey non-response. *International Statistical
    Review*, 78(1), 40-64.
    https://doi.org/10.1111/j.1751-5823.2010.00103.x
    """
    validate_dataframe(df, param="df")
    if direction not in _DIRECTIONS:
        raise ValueError(
            "direction must be one of 'forward', 'backward', 'nearest', "
            f"'forward_then_backward' or 'closest'; got {direction!r}"
        )
    by_cols = _normalize_by(df, by)
    df = _normalize_missing(df)
    result = df.copy()
    classes = _class_rows(df, by_cols)
    legacy = direction == "nearest"
    method = "forward_then_backward" if legacy else direction
    differs_from_closest = False

    for col in df.columns:
        if col in by_cols or not result[col].isna().any():
            continue
        observed = df[col].notna().to_numpy()
        donors = _sequence_donors(observed, classes, method)
        if legacy:
            closest = _sequence_donors(observed, classes, "closest")
            differs_from_closest = differs_from_closest or bool(
                (donors != closest).any()
            )
        take = np.where(donors >= 0, donors, np.arange(len(df)))
        result[col] = df[col].iloc[take].set_axis(df.index)

    if differs_from_closest:
        warnings.warn(
            "direction='nearest' fills an interior gap with the previous observed "
            "value even when the next one is closer; a future release will make it "
            "pick the closest observed value. Pass direction='forward_then_backward' "
            "to keep the current result or direction='closest' for the nearest "
            "observed value.",
            FutureWarning,
            stacklevel=2,
        )
    return result


def impute_hotdeck_weighted(
    df: pd.DataFrame,
    n_donors: int = 5,
    random_state: int | None = None,
    by: ByType = None,
) -> pd.DataFrame:
    """Impute missing values via distance-weighted hot-deck.

    For each missing cell, all other columns are used as matching
    variables to compute Gower distances between the recipient row and
    all candidate donor rows.  The ``n_donors`` nearest donors form the
    pool and one of them is sampled with probability proportional to
    ``1 / (1 + distance)``, preserving stochasticity while favouring
    similar records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_donors : int, default 5
        Candidate donor pool size: the *n_donors* nearest donors by
        Gower distance are identified (equally near donors are ordered at
        random), then one is sampled with the weights above.  Larger values
        increase randomness; ``n_donors=1`` is nearest-neighbour hot-deck
        and is deterministic unless several donors are equally near, in
        which case one of them is chosen at random.
    random_state : int, numpy integer, Generator, RandomState or None, default None
        Seed for the random number generator.
        Accepts ``None``, a non-negative integer of any size (Python or numpy),
        a ``numpy.random.Generator`` or a ``numpy.random.RandomState``; any other
        value raises :class:`~missingly.exceptions.InvalidSeedError`.  A boolean
        is deprecated: it gives a :class:`FutureWarning` and acts as ``1`` or ``0``.
    by : str, list of str or None, default None
        Column(s) that define the imputation classes.  Donors come only from
        the recipient's class; the ``by`` columns are neither imputed nor
        used as matching variables.  Rows with a missing class key form a
        class of their own.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by observed values
        drawn from distance-weighted donor rows.

    Raises
    ------
    InvalidSeedError
        If ``random_state`` is not an accepted seed type.
    TypeError
        If *df* is not a :class:`pandas.DataFrame`, if *by* is not a column
        name, a list of names or ``None``, or if *n_donors* is not an integer.
    ValueError
        If *df* is empty or *n_donors* is not positive.
    ValueError
        If a column (or class) has a missing value but no observed donor, or
        a ``by`` label is not a column.

    Warns
    -----
    FutureWarning
        If ``random_state`` is a boolean; it is treated as ``1`` or ``0``.
    UserWarning
        If the frame has more than ``config.large_df_threshold`` rows or the
        number of recipient-donor pairs is very large.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     "age":  [25.0, np.nan, 35.0, 40.0],
    ...     "income": [30000.0, 45000.0, np.nan, 60000.0],
    ... })
    >>> impute_hotdeck_weighted(df, n_donors=1, random_state=0)
        age   income
    0  25.0  30000.0
    1  40.0  45000.0
    2  35.0  60000.0
    3  40.0  60000.0

    Notes
    -----
    Donor-selection rule: for a recipient row *i* and every donor row *d*
    (rows of the same class with the target column observed), the Gower
    distance ``D(i, d)`` is the mean over the matching variables observed in
    both rows of ``|x_i - x_d| / range`` (numeric, datetime and timedelta
    columns; the range is the observed range of the whole column) or of
    ``0`` / ``1`` for equal / different values (categorical, bool and
    ``boolean`` columns); a pair with no usable matcher has distance 1.  The
    pool holds the ``n_donors`` donors with the smallest ``D`` (ties broken
    at random) and donor *d* of the pool is drawn with probability
    ``(1 / (1 + D(i, d))) / sum over the pool of 1 / (1 + D)``.  The weight is
    bounded between 1/2 and 1, so an exact match is at most twice as likely
    as the farthest donor and the randomness controlled by ``n_donors``
    is retained.  Imputed values are always observed values of the donor
    rows.

    This is a stochastic nearest-neighbour hot-deck following the
    distance-based donor selection reviewed by Andridge and Little (2010).
    It is not a port of R's ``VIM::hotdeck()`` (a sequential or random
    hot-deck within domain variables) and not SAS fractional imputation,
    which keeps several weighted donors per recipient.

    Gower distance handles mixed numeric and categorical columns natively
    without ordinal encoding.  Donor values are copied exactly, so ``Int64``
    integers keep full precision and every column keeps its dtype.
    Distances are computed with NumPy in chunks; the cost is
    O(n_missing × n_donors × n_features) per column.  A ``UserWarning`` is
    emitted for frames with more than ``config.large_df_threshold`` rows or
    with a very large number of recipient-donor pairs.

    References
    ----------
    Gower, J.C. (1971). A general coefficient of similarity and some
    of its properties. *Biometrics*, 27(4), 857-871.
    Andridge, R.R. & Little, R.J.A. (2010). A review of hot deck
    imputation for survey non-response. *International Statistical
    Review*, 78(1), 40-64.
    https://doi.org/10.1111/j.1751-5823.2010.00103.x
    """
    validate_dataframe(df, param="df")
    validate_positive_int(n_donors, param="n_donors")
    by_cols = _normalize_by(df, by)

    df = _normalize_missing(df)
    classes = _class_rows(df, by_cols)
    _warn_if_large(len(df), _pair_count(df, classes))
    rng = as_generator(random_state)
    result = df.copy()
    features = [_build_feature(df[col]) for col in df.columns]

    for col_idx, col in enumerate(df.columns):
        if col in by_cols:
            continue
        missing = df[col].isna().to_numpy()
        if not missing.any():
            continue
        matchers = _matching_features(features, df.columns, col_idx, by_cols)

        def draw(recipients: np.ndarray, donors: np.ndarray) -> np.ndarray:
            return _weighted_donor_rows(matchers, recipients, donors, n_donors, rng)

        source = _draw_sources(df, col, by_cols, classes, draw)
        # Copy the donors' own values so the column keeps its dtype exactly.
        result.iloc[np.flatnonzero(missing), col_idx] = (
            df[col].iloc[source[missing]].to_numpy()
        )
        result[col] = _restore_categorical_dtype(result[col], df[col].dtype)

    return result
