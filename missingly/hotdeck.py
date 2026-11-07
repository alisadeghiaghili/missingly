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
    in that column.  Equivalent to SAS ``PROC SURVEYIMPUTE METHOD=SRS``
    and SPSS ``MVA / HOTDECK``.

``impute_hotdeck_sequential``
    *Sequential (carry-forward / carry-backward) hot-deck*: scan the
    column in order and substitute the closest preceding observed value
    (``direction='forward'``), the closest following observed value
    (``direction='backward'``), or try forward first and fall back to
    backward when the column starts with NaN
    (``direction='nearest'``, the default).  Common in time-ordered
    survey data and panel datasets.

``impute_hotdeck_weighted``
    *Distance-weighted hot-deck*: for each missing row, compute Gower
    distances to all donor rows using all *other* columns as matching
    variables, then sample a donor with probability inversely proportional
    to distance.  This is the closest Python equivalent to VIM's
    ``hotdeck()`` and SAS ``PROC SURVEYIMPUTE METHOD=FRAC``.
    Complexity is O(n_missing × n_donors) per column, so a
    ``UserWarning`` is emitted when the DataFrame exceeds
    ``config.large_df_threshold`` rows.

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
from typing import Literal

import numpy as np
import pandas as pd

from missingly._validation import validate_dataframe, validate_positive_int
from missingly.config import config as _config
from missingly.impute import _normalize_missing, _restore_categorical_dtype


__all__ = [
    "impute_hotdeck_random",
    "impute_hotdeck_sequential",
    "impute_hotdeck_weighted",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gower_distances_to_donors(
    row_miss: np.ndarray,
    X_donors: np.ndarray,
    col_is_numeric: np.ndarray,
    col_ranges: np.ndarray,
) -> np.ndarray:
    """Compute Gower distance from one missing row to all donor rows.

    Parameters
    ----------
    row_miss : np.ndarray of shape (n_features,)
        Feature vector of the recipient row (may contain NaN for the
        target column but feature columns should be filled externally).
    X_donors : np.ndarray of shape (n_donors, n_features)
        Feature matrix of donor rows.
    col_is_numeric : np.ndarray of bool, shape (n_features,)
        True for numeric columns.
    col_ranges : np.ndarray of float, shape (n_features,)
        Per-column range (max - min) for numeric normalisation.
        Zero-range columns contribute 0 to the distance.

    Returns
    -------
    np.ndarray of shape (n_donors,)
        Gower distance in [0, 1] for each donor row.
    """
    n_donors, n_features = X_donors.shape
    numerator = np.zeros(n_donors, dtype=np.float64)
    denominator = np.zeros(n_donors, dtype=np.float64)

    for j in range(n_features):
        r_j = row_miss[j]
        d_j = X_donors[:, j]

        # Skip features where the recipient itself is missing
        if bool(pd.isna(r_j)):
            continue

        donor_observed = np.fromiter(
            (not bool(pd.isna(value)) for value in d_j),
            dtype=bool,
            count=n_donors,
        )
        if not donor_observed.any():
            continue

        if col_is_numeric[j]:
            rng = col_ranges[j]
            if rng == 0.0:
                contribution = np.zeros(n_donors, dtype=np.float64)
            else:
                donor_numeric = pd.to_numeric(
                    pd.Series(d_j), errors="coerce"
                ).to_numpy(dtype=np.float64)
                contribution = np.abs(donor_numeric - float(r_j)) / rng
        else:
            contribution = np.fromiter(
                (0.0 if value == r_j else 1.0 for value in d_j),
                dtype=np.float64,
                count=n_donors,
            )

        # Missing-value handling is pairwise: donors that do not have this
        # matcher must not receive a fixed penalty or share a denominator with
        # donors for which it is observed.
        numerator[donor_observed] += contribution[donor_observed]
        denominator[donor_observed] += 1.0

    return np.divide(
        numerator,
        denominator,
        out=np.ones(n_donors, dtype=np.float64),
        where=denominator > 0.0,
    )


def _gower_numeric_mask(df: pd.DataFrame) -> np.ndarray:
    """Return which columns Gower treats as real-valued.

    Bool and nullable ``boolean`` columns are numeric for pandas but cannot be
    subtracted or ranged, so they are compared as categories instead.

    Parameters
    ----------
    df : pd.DataFrame
        Frame whose columns are classified.

    Returns
    -------
    np.ndarray of bool, shape (n_columns,)
        True for numeric columns that are not boolean.
    """
    return np.array(
        [
            pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
            for col in df.columns
        ],
        dtype=bool,
    )


def _gower_column_ranges(
    df: pd.DataFrame,
    col_is_numeric: np.ndarray,
) -> np.ndarray:
    """Return the observed range (max - min) of every numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col_is_numeric : np.ndarray of bool
        Output of :func:`_gower_numeric_mask`.

    Returns
    -------
    np.ndarray of float, shape (n_columns,)
        Range per numeric column; 0.0 for other columns and for numeric
        columns without an observed value.
    """
    col_ranges = np.zeros(len(df.columns), dtype=np.float64)
    for j, col in enumerate(df.columns):
        if col_is_numeric[j]:
            lo, hi = df[col].min(), df[col].max()
            col_ranges[j] = float(hi - lo) if pd.notna(lo) and pd.notna(hi) else 0.0
    return col_ranges


def _gower_matrix(
    df: pd.DataFrame,
    col_is_numeric: np.ndarray,
) -> np.ndarray:
    """Return the object matrix used for Gower matching.

    Numeric columns hold their values as floats (``pd.NA`` for missing), all
    other columns keep their original values for equality matching.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col_is_numeric : np.ndarray of bool
        Output of :func:`_gower_numeric_mask`.

    Returns
    -------
    np.ndarray of object, shape (n_rows, n_columns)
        Matching matrix; each row is one record.
    """
    matrix = df.to_numpy(dtype=object).copy()
    for j, col in enumerate(df.columns):
        if col_is_numeric[j]:
            matrix[:, j] = pd.to_numeric(df[col], errors="coerce")
    return matrix


def _pick_inverse_distance_donor(
    dists: np.ndarray,
    n_donors: int,
    rng: np.random.Generator,
) -> int:
    """Sample one of the *n_donors* nearest donors with inverse-distance weights.

    Parameters
    ----------
    dists : np.ndarray of float
        Gower distance from the recipient to every candidate donor.
    n_donors : int
        Size of the candidate pool.
    rng : np.random.Generator
        Random number generator used for the draw.

    Returns
    -------
    int
        Position in *dists* of the sampled donor.
    """
    k = min(n_donors, len(dists))
    if k < len(dists):
        top_k_idx = np.argpartition(dists, k - 1)[:k]
    else:
        top_k_idx = np.arange(len(dists))
    top_k_dists = dists[top_k_idx]
    # Inverse-distance weights; ties / zero distances fall back to uniform.
    inv = 1.0 / (top_k_dists + 1e-10)
    weights = inv / inv.sum()
    return int(top_k_idx[rng.choice(len(top_k_idx), p=weights)])


def _weighted_donor_rows(
    matrix: np.ndarray,
    target_idx: int,
    missing_rows: np.ndarray,
    donor_rows: np.ndarray,
    col_is_numeric: np.ndarray,
    col_ranges: np.ndarray,
    n_donors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick one donor row for every missing row of one target column.

    Parameters
    ----------
    matrix : np.ndarray of object
        Matching matrix from :func:`_gower_matrix`.
    target_idx : int
        Position of the column being imputed; excluded from the matching.
    missing_rows : np.ndarray of int
        Rows whose value in the target column is missing.
    donor_rows : np.ndarray of int
        Rows whose value in the target column is observed.
    col_is_numeric : np.ndarray of bool
        Output of :func:`_gower_numeric_mask`.
    col_ranges : np.ndarray of float
        Output of :func:`_gower_column_ranges`.
    n_donors : int
        Candidate pool size.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of int
        Donor row (position in the frame) for each missing row, in order.
    """
    feat_idx = [j for j in range(matrix.shape[1]) if j != target_idx]
    donors = matrix[np.ix_(donor_rows, feat_idx)].astype(object)
    feat_is_num = col_is_numeric[feat_idx]
    feat_ranges = col_ranges[feat_idx]
    chosen = np.empty(len(missing_rows), dtype=np.intp)
    for i, miss_row in enumerate(missing_rows):
        row_feat = matrix[miss_row, feat_idx].copy()
        dists = _gower_distances_to_donors(row_feat, donors, feat_is_num, feat_ranges)
        chosen[i] = donor_rows[_pick_inverse_distance_donor(dists, n_donors, rng)]
    return chosen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def impute_hotdeck_random(
    df: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Impute missing values via simple random hot-deck.

    For each column that contains missing values, a donor is drawn
    uniformly at random (with replacement) from the rows that have an
    observed value in that column.  The procedure is column-independent:
    each column is imputed separately.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    random_state : int or None, default None
        Seed for the random number generator.  Pass an integer for
        reproducible results.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by randomly sampled
        observed values from the same column.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ValueError
        If any column is entirely missing (no donor available).

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

    Notes
    -----
    This method is equivalent to SAS ``PROC SURVEYIMPUTE METHOD=SRS``
    and SPSS ``MVA / HOTDECK``.  Because donors are sampled with
    replacement the marginal distribution of each column is preserved
    in expectation, but variance may be underestimated for small donor
    pools.  For variance-correct inference, use multiple imputation
    (``impute_mice`` with ``n_imputations > 1``) instead.

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
    df = _normalize_missing(df)
    rng = np.random.default_rng(random_state)
    result = df.copy()

    for col in df.columns:
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue
        observed = df.loc[~missing_mask, col]
        if len(observed) == 0:
            raise ValueError(
                f"Column {col!r} has no observed values; cannot hot-deck impute."
            )
        picks = rng.choice(len(observed), size=missing_mask.sum(), replace=True)
        result.loc[missing_mask, col] = observed.iloc[picks].to_numpy()
        result[col] = _restore_categorical_dtype(result[col], df[col].dtype)

    return result


def impute_hotdeck_sequential(
    df: pd.DataFrame,
    direction: Literal["forward", "backward", "nearest"] = "nearest",
) -> pd.DataFrame:
    """Impute missing values via sequential (carry-forward/backward) hot-deck.

    Scans each column in row order and fills each missing cell with the
    nearest observed value in the specified direction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    direction : {"forward", "backward", "nearest"}, default ``"nearest"``
        Direction in which to search for a donor:

        * ``"forward"``  — use the last seen observed value (LOCF).
          Leaves leading NaN unfilled.
        * ``"backward"`` — use the next observed value (NOCB).
          Leaves trailing NaN unfilled.
        * ``"nearest"``  — try forward first, then backward, so that
          no NaN remains as long as the column has at least one
          observed value.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values filled by carry-forward /
        carry-backward substitution.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    ValueError
        If *direction* is not one of the supported values.

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
    >>> impute_hotdeck_sequential(df, direction="nearest")
       score
    0   10.0
    1   10.0
    2   10.0
    3   30.0

    Notes
    -----
    Forward fill (LOCF) is equivalent to pandas ``ffill()`` and is
    common in time-ordered panel datasets where the last observed
    value is the best proxy for the next period.  It assumes data are
    MAR or MCAR; using LOCF under MNAR inflates bias.
    """
    validate_dataframe(df, param="df")
    if direction not in {"forward", "backward", "nearest"}:
        raise ValueError(
            f"direction must be 'forward', 'backward', or 'nearest'; got {direction!r}"
        )
    df = _normalize_missing(df)
    result = df.copy()

    for col in df.columns:
        if not result[col].isna().any():
            continue
        orig_dtype = df[col].dtype
        if direction == "forward":
            filled = result[col].ffill()
        elif direction == "backward":
            filled = result[col].bfill()
        else:  # nearest: forward first, then backward to catch leading NaN
            filled = result[col].ffill().bfill()
        result[col] = _restore_categorical_dtype(filled, orig_dtype)

    return result


def impute_hotdeck_weighted(
    df: pd.DataFrame,
    n_donors: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Impute missing values via distance-weighted hot-deck.

    For each missing cell, all other columns are used as matching
    variables to compute Gower distances between the recipient row and
    all candidate donor rows.  A donor is then sampled with probability
    inversely proportional to its distance, preserving stochasticity
    while favouring similar records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_donors : int, default 5
        Candidate donor pool size: the *n_donors* nearest donors by
        Gower distance are identified, then one is sampled with
        inverse-distance weights.  Larger values increase randomness;
        ``n_donors=1`` is deterministic nearest-neighbour hot-deck.
    random_state : int or None, default None
        Seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values replaced by observed values
        drawn from distance-weighted donor rows.

    Raises
    ------
    TypeError
        If *df* is not a :class:`pandas.DataFrame`.
    ValueError
        If *df* is empty.
    TypeError
        If *n_donors* is not a positive integer.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     "age":  [25.0, np.nan, 35.0, 40.0],
    ...     "income": [30000.0, 45000.0, np.nan, 60000.0],
    ... })
    >>> impute_hotdeck_weighted(df, n_donors=2, random_state=0)
        age   income
    0  25.0  30000.0
    1  40.0  45000.0
    2  35.0  60000.0
    3  40.0  60000.0

    Notes
    -----
    This method is the closest Python equivalent to R's ``VIM::hotdeck()``
    and SAS ``PROC SURVEYIMPUTE METHOD=FRAC``.  Gower distance handles
    mixed numeric and categorical columns natively without ordinal
    encoding.  bool and nullable ``boolean`` columns are matched as
    categories.  Donor values are copied exactly, so ``Int64`` integers
    keep full precision and every column keeps its dtype.
    For large DataFrames (> ``config.large_df_threshold`` rows)
    a ``UserWarning`` is emitted because complexity is
    O(n_missing × n_donors) per column.

    References
    ----------
    Gower, J.C. (1971). A general coefficient of similarity and some
    of its properties. *Biometrics*, 27(4), 857-871.
    Andridge & Little (2010), ibid.
    """
    validate_dataframe(df, param="df")
    validate_positive_int(n_donors, param="n_donors")

    if len(df) > _config.large_df_threshold:
        warnings.warn(
            f"impute_hotdeck_weighted: DataFrame has {len(df):,} rows. "
            "Weighted hot-deck is O(n_missing × n_donors) per column and may "
            "be slow. Consider impute_hotdeck_random() for large datasets.",
            UserWarning,
            stacklevel=2,
        )

    df = _normalize_missing(df)
    rng = np.random.default_rng(random_state)
    result = df.copy()

    col_is_numeric = _gower_numeric_mask(df)
    col_ranges = _gower_column_ranges(df, col_is_numeric)
    matrix = _gower_matrix(df, col_is_numeric)

    for col_idx, col in enumerate(df.columns):
        missing_rows = np.where(df[col].isna())[0]
        if len(missing_rows) == 0:
            continue

        donor_rows = np.where(~df[col].isna())[0]
        if len(donor_rows) == 0:
            raise ValueError(
                f"Column {col!r} has no observed values; cannot hot-deck impute."
            )

        chosen = _weighted_donor_rows(
            matrix,
            col_idx,
            missing_rows,
            donor_rows,
            col_is_numeric,
            col_ranges,
            n_donors,
            rng,
        )
        # Copy the donors' own values so the column keeps its dtype exactly.
        result.iloc[missing_rows, col_idx] = df[col].iloc[chosen].to_numpy()
        result[col] = _restore_categorical_dtype(result[col], df[col].dtype)

    return result
