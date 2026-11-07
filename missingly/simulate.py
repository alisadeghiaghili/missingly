"""Missing data simulation utilities."""

from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

_VALID_TAILS = ("upper", "lower")
# Odds weights (beyond-threshold side, other side) of the two-group models.
_MAR_WEIGHTS = (2.0, 0.5)
_MNAR_WEIGHTS = (2.0, 0.25)
_CALIBRATION_STEPS = 80


def _with_nullable_bools(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return a copy of *df* whose plain bool *columns* are nullable ``boolean``.

    NumPy ``bool`` columns cannot hold ``NaN``; pandas refuses or warns when
    masking them, so they are converted before any cell is blanked.
    """
    result = df.copy()
    for col in columns:
        is_plain_bool = pd.api.types.is_bool_dtype(result[col])
        if is_plain_bool and not pd.api.types.is_extension_array_dtype(result[col]):
            result[col] = result[col].astype("boolean")
    return result


def _default_columns(df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
    """Return *columns*, or every numeric column of *df* when it is None."""
    if columns is not None:
        return columns
    return df.select_dtypes(include="number").columns.tolist()


def _check_tail_and_frac(tail: str, frac: float) -> None:
    """Raise ``ValueError`` when *tail* or *frac* is not usable."""
    if tail not in _VALID_TAILS:
        raise ValueError(f"tail must be one of {sorted(_VALID_TAILS)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")


def _high_share(
    values: np.ndarray, tail: str, threshold: Optional[float]
) -> np.ndarray:
    """Return, per row, the share of the high-probability group it belongs to.

    With an explicit *threshold* the share is 1 for values strictly beyond it
    (above for ``tail="upper"``, below for ``"lower"``) and 0 otherwise. By
    default the rows are split at the median by rank: values strictly beyond
    the median get 1, values strictly before it get 0 and the tied values at
    the boundary share the remaining half fractionally, so the shares sum to
    half of the rows even for binary and discrete data. Missing values get 0.
    """
    x = values if tail == "upper" else -values
    valid = ~np.isnan(x)
    share = np.zeros(len(x))
    if threshold is not None:
        cut = threshold if tail == "upper" else -threshold
        share[valid] = x[valid] > cut
        return share
    levels, inverse, counts = np.unique(
        x[valid], return_inverse=True, return_counts=True
    )
    n_greater = counts.sum() - np.cumsum(counts)
    level_share = np.clip((counts.sum() / 2 - n_greater) / counts, 0.0, 1.0)
    share[valid] = level_share[inverse]
    return share


def _calibrated_probabilities(
    share: np.ndarray,
    eligible: np.ndarray,
    frac: float,
    weights: tuple,
) -> np.ndarray:
    """Return ``min(1, c * w_i)`` with ``c`` solved so the mean over eligible rows is *frac*.

    ``w_i = w_low + share_i * (w_high - w_low)``; rows that are not eligible
    get probability 0. The scale ``c`` is found by bisection on the monotone
    mean, which stays exact when some probabilities are capped at 1.
    """
    w_high, w_low = weights
    odds = np.where(eligible, w_low + share * (w_high - w_low), 0.0)
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return np.zeros(len(odds))
    levels, counts = np.unique(odds[eligible], return_counts=True)
    target = frac * n_eligible
    low, high = 0.0, 1.0 / levels.min()
    for _ in range(_CALIBRATION_STEPS):
        mid = (low + high) / 2
        if (counts * np.minimum(1.0, mid * levels)).sum() < target:
            low = mid
        else:
            high = mid
    return np.minimum(1.0, (low + high) / 2 * odds)


def _biased_missing(
    df: pd.DataFrame,
    target_col: str,
    driver_col: str,
    frac: float,
    tail: str,
    threshold: Optional[float],
    weights: tuple,
    random_state: Optional[int],
) -> pd.DataFrame:
    """Blank *target_col* with probabilities driven by *driver_col* (MAR / MNAR)."""
    _check_tail_and_frac(tail, frac)
    rng = np.random.default_rng(random_state)
    result = _with_nullable_bools(df, [target_col])
    driver = pd.to_numeric(df[driver_col], errors="coerce").to_numpy(dtype=float)
    eligible = ~result[target_col].isnull().to_numpy()
    share = _high_share(driver, tail, threshold)
    if threshold is not None and eligible.any() and np.unique(share[eligible]).size < 2:
        warnings.warn(
            f"threshold {threshold!r} puts every row of {driver_col!r} on the same "
            "side, so the missingness is MCAR; choose a threshold inside the data "
            "range.",
            UserWarning,
            stacklevel=4,
        )
    prob = _calibrated_probabilities(share, eligible, frac, weights)
    result.loc[rng.random(len(df)) < prob, target_col] = np.nan
    return result


def simulate_mcar(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce MCAR (Missing Completely At Random) missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Complete input frame (no existing missing values).
    frac : float, default 0.1
        Fraction of rows blanked in each target column, strictly between 0
        and 1.
    columns : list of str, optional
        Columns to blank. Defaults to every numeric column; bool columns are
        only blanked when named here.
    random_state : int or None, default None
        Seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values introduced.

    Raises
    ------
    ValueError
        If *df* already contains missing values, *frac* is outside (0, 1),
        or a name in *columns* is not in *df*.

    Notes
    -----
    Each target column loses ``max(1, round(n * frac))`` values. bool columns
    are returned as nullable ``boolean`` so they can hold missing values.
    int64 columns are returned as float64, as in earlier releases.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    >>> int(simulate_mcar(df, 0.5, random_state=0)["x"].isna().sum())
    2
    >>> flags = pd.DataFrame({"flag": [True, False, True, False]})
    >>> str(simulate_mcar(flags, 0.5, columns=["flag"], random_state=0)["flag"].dtype)
    'boolean'
    """
    if df.isnull().any().any():
        raise ValueError(
            "simulate_mcar expects a complete DataFrame (no existing missing values). "
            "Fill or drop NaNs before calling this function."
        )
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}.")
    rng = np.random.default_rng(random_state)
    target_cols = _default_columns(df, columns)
    result = _with_nullable_bools(df, target_cols)
    for col in target_cols:
        n = len(result)
        n_missing = max(1, int(round(n * frac)))
        indices = rng.choice(n, size=n_missing, replace=False)
        result.iloc[indices, result.columns.get_loc(col)] = np.nan
    return result


def simulate_mar(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    predictor_col: str,
    tail: str = "upper",
    threshold: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce MAR (Missing At Random) missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Complete input frame (no existing missing values).
    frac : float, default 0.1
        Target proportion of *target_col* that is blanked, strictly between 0
        and 1. It is the expected overall rate: the realised count is binomial
        around it.
    target_col : str
        Column to blank.
    predictor_col : str
        Numeric column that drives the missingness probability. It must differ
        from *target_col*.
    tail : {"upper", "lower"}, default "upper"
        Rows on this side of *threshold* are blanked more often.
    threshold : float, optional
        Split point for *predictor_col*; rows strictly beyond it form the
        high-probability group. Defaults to the median, split by rank (see
        Notes), which also works for binary and discrete drivers.
    random_state : int or None, default None
        Seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values introduced in *target_col*.

    Raises
    ------
    ValueError
        If *df* already contains missing values, *tail* or *frac* is invalid,
        *predictor_col* is not numeric, or *predictor_col* equals *target_col*
        (the missingness would then depend on the blanked value, which is
        MNAR).
    KeyError
        If *target_col* or *predictor_col* is not in *df*.

    Warns
    -----
    UserWarning
        If an explicit *threshold* leaves every row on one side, so the
        result is MCAR.

    Notes
    -----
    Each row is blanked independently with probability
    ``p_i = min(1, c * w_i)``. The odds weight ``w_i = 0.5 + 1.5 * s_i`` mixes
    the low weight 0.5 and the high weight 2 by the share ``s_i`` of the
    high-probability group the row belongs to. ``s_i`` is a function of the
    observed ``predictor_col`` only, so the blanked value adds no information
    once the driver is known (MAR). The scale ``c`` is solved so the mean of
    ``p_i`` equals *frac*, as ``mice::ampute`` solves its intercept for
    ``prop`` (Schouten, Lugtig and Vink, 2018, J. Stat. Comput. Simul.
    88(15)). Without *threshold* the rows are split by rank at the median:
    values strictly beyond it have ``s_i = 1``, values before it ``0`` and
    tied values at the boundary share the remaining half of the rows, so a
    binary driver with 80% ones gives the ones ``s = 0.625``. With a
    *threshold*, ``s_i`` is 1 for values strictly beyond it and 0 otherwise.
    Earlier releases used a strict ``> median`` rule that gave every row the
    same probability on such drivers, and unnormalised probabilities that
    produced 1.25 times *frac*. bool columns are returned as nullable
    ``boolean`` so they can hold missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    >>> out = simulate_mar(df, 0.5, target_col="y", predictor_col="x", random_state=0)
    >>> bool(out["y"].isna().any())
    True
    >>> simulate_mar(df, 0.5, target_col="y", predictor_col="y")
    Traceback (most recent call last):
        ...
    ValueError: predictor_col must differ from target_col; got 'y' for both (the missingness would depend on the value being blanked, which is MNAR, not MAR).
    """
    if df.isnull().any().any():
        raise ValueError(
            "simulate_mar expects a complete DataFrame (no existing missing values). "
            "Fill or drop NaNs before calling this function."
        )
    _check_tail_and_frac(tail, frac)
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if predictor_col not in df.columns:
        raise KeyError(f"predictor_col {predictor_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[predictor_col]):
        raise ValueError(f"predictor_col {predictor_col!r} must be numeric.")
    return _simulate_mar_raw(
        df,
        frac,
        target_col=target_col,
        predictor_col=predictor_col,
        tail=tail,
        threshold=threshold,
        random_state=random_state,
    )


def simulate_mnar(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    threshold: Optional[float] = None,
    tail: str = "upper",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce MNAR (Missing Not At Random) missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Complete input frame (no existing missing values).
    frac : float, default 0.1
        Target proportion of *target_col* that is blanked, strictly between 0
        and 1 (expected overall rate; the realised count is binomial).
    target_col : str
        Numeric column whose own values drive its missingness.
    threshold : float, optional
        Split point for *target_col*; values strictly beyond it form the
        high-probability group. Defaults to the median, split by rank (see
        Notes), which also works for discrete columns.
    tail : {"upper", "lower"}, default "upper"
        Values on this side of *threshold* are blanked more often.
    random_state : int or None, default None
        Seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing values introduced in *target_col*.

    Raises
    ------
    ValueError
        If *tail* or *frac* is invalid, or *target_col* is not numeric.
    KeyError
        If *target_col* is not in *df*.

    Warns
    -----
    UserWarning
        If an explicit *threshold* leaves every value on one side, so the
        result is MCAR.

    Notes
    -----
    Each row is blanked independently with probability
    ``p_i = min(1, c * w_i)`` with odds weight ``w_i = 0.25 + 1.75 * s_i``,
    where ``s_i`` is the share of the high-probability group the row's *own*
    value belongs to (rank split at the median with fractional sharing of tied
    values, or strict comparison with *threshold*; see ``simulate_mar``).
    Because ``s_i`` depends on the value that is blanked, the data are MNAR.
    The scale ``c`` is solved so the mean of ``p_i`` equals *frac*. Values
    that are already missing get probability 0 and *frac* then refers to the
    still-observed rows. bool columns are returned as nullable ``boolean`` so
    they can hold missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    >>> out = simulate_mnar(df, 0.5, target_col="y", random_state=0)
    >>> bool(out["y"].isna().any())
    True
    """
    _check_tail_and_frac(tail, frac)
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"target_col {target_col!r} must be numeric for MNAR simulation."
        )
    return _simulate_mnar_raw(
        df,
        frac,
        target_col=target_col,
        threshold=threshold,
        tail=tail,
        random_state=random_state,
    )


def simulate_mixed(
    df: pd.DataFrame,
    mechanisms: List[Dict],
    *,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Apply a mixture of MCAR / MAR / MNAR mechanisms sequentially.

    Each dict in *mechanisms* must have a ``"mechanism"`` key with value
    ``"mcar"``, ``"mar"``, or ``"mnar"`` (case-insensitive).

    Unlike the individual simulate_* functions, simulate_mixed allows
    each step to operate on a DataFrame that already has some NaN values
    from previous steps — the per-step NaN guard is deliberately skipped.

    A MAR step is only valid while its ``predictor_col`` stays observed, so
    the predictor columns of all MAR steps are protected: a step that names
    one as its target (or in an MCAR step's ``columns``) raises, and an MCAR
    step without ``columns`` skips them.

    Parameters
    ----------
    df : pd.DataFrame
        Complete input frame (no existing missing values).
    mechanisms : list of dict
        One dict per step: the ``"mechanism"`` key plus the keyword arguments
        of that mechanism (for example ``frac`` or ``target_col``).
    random_state : int or None, default None
        Base seed; step *i* uses ``random_state + i``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the missingness of every step applied.

    Raises
    ------
    ValueError
        If *df* already contains missing values, a mechanism is unknown, a MAR
        step uses its own target as ``predictor_col``, or a step would blank
        the ``predictor_col`` of a MAR step.

    Notes
    -----
    MAR and MNAR steps blank ``frac`` of the rows that are still observed in
    their target column, with the probabilities described in
    ``simulate_mar`` / ``simulate_mnar``. bool columns are returned as nullable ``boolean`` so they can hold missing
    values.
    """
    if df.isnull().any().any():
        raise ValueError(
            "simulate_mixed expects a complete DataFrame (no existing missing values). "
            "Fill or drop NaNs before calling this function."
        )

    _sim_dispatch = {
        "mcar": _simulate_mcar_raw,
        "mar": _simulate_mar_raw,
        "mnar": _simulate_mnar_raw,
    }

    drivers = _protected_drivers(mechanisms)
    result = df.copy()
    for i, spec in enumerate(mechanisms):
        spec = dict(spec)
        mtype = spec.pop("mechanism", None)
        if mtype is None or mtype.lower() not in _sim_dispatch:
            raise ValueError(
                f"unknown mechanism {mtype!r}. Choose from {sorted(_sim_dispatch)}."
            )
        if mtype.lower() == "mcar" and spec.get("columns") is None:
            spec["columns"] = [
                c for c in _default_columns(df, None) if c not in drivers
            ]
        seed = (random_state + i) if random_state is not None else None
        result = _sim_dispatch[mtype.lower()](result, random_state=seed, **spec)
    return result


# ---------------------------------------------------------------------------
# Internal "raw" helpers used by simulate_mixed (no existing-NaN guard)
# ---------------------------------------------------------------------------


def _protected_drivers(mechanisms: List[Dict]) -> set:
    """Return the MAR drivers of *mechanisms*; raise if a step would mask one.

    A MAR step is only valid while its ``predictor_col`` stays observed, so no
    step may blank that column, whatever the step order.
    """
    steps = [m for m in mechanisms if isinstance(m, dict)]
    mar_steps = [m for m in steps if str(m.get("mechanism", "")).lower() == "mar"]
    drivers = {m["predictor_col"] for m in mar_steps if "predictor_col" in m}
    for step in steps:
        kind = str(step.get("mechanism", "")).lower()
        if kind == "mcar":
            masked = step.get("columns") or []
        else:
            masked = [step["target_col"]] if "target_col" in step else []
        clash = sorted(c for c in masked if c in drivers)
        if clash:
            raise ValueError(
                f"column {clash[0]!r} is the predictor_col (driver) of a MAR step "
                "and cannot be blanked by another step; the MAR mechanism would "
                "no longer depend on observed values."
            )
    return drivers


def _simulate_mcar_raw(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    rng = np.random.default_rng(random_state)
    target_cols = _default_columns(df, columns)
    result = _with_nullable_bools(df, target_cols)
    for col in target_cols:
        available = result[col].dropna().index
        if len(available) == 0:
            continue
        n_missing = min(max(1, int(round(len(df) * frac))), len(available))
        chosen = rng.choice(len(available), size=n_missing, replace=False)
        result.loc[available[chosen], col] = np.nan
    return result


def _simulate_mar_raw(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    predictor_col: str,
    tail: str = "upper",
    threshold: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    if predictor_col == target_col:
        raise ValueError(
            f"predictor_col must differ from target_col; got {predictor_col!r} for "
            "both (the missingness would depend on the value being blanked, which "
            "is MNAR, not MAR)."
        )
    return _biased_missing(
        df, target_col, predictor_col, frac, tail, threshold, _MAR_WEIGHTS, random_state
    )


def _simulate_mnar_raw(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    threshold: Optional[float] = None,
    tail: str = "upper",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    return _biased_missing(
        df, target_col, target_col, frac, tail, threshold, _MNAR_WEIGHTS, random_state
    )
