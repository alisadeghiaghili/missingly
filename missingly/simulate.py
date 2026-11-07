"""Missing data simulation utilities."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


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
        Base missingness fraction, strictly between 0 and 1.
    target_col : str
        Column to blank.
    predictor_col : str
        Numeric column that drives the missingness probability.
    tail : {"upper", "lower"}, default "upper"
        Rows on this side of *threshold* are blanked more often.
    threshold : float, optional
        Split point for *predictor_col*. Defaults to its median.
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
        or *predictor_col* is not numeric.
    KeyError
        If *target_col* or *predictor_col* is not in *df*.

    Notes
    -----
    bool columns are returned as nullable ``boolean`` so they can hold missing
    values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    >>> out = simulate_mar(df, 0.5, target_col="y", predictor_col="x", random_state=0)
    >>> bool(out["y"].isna().any())
    True
    """
    if df.isnull().any().any():
        raise ValueError(
            "simulate_mar expects a complete DataFrame (no existing missing values). "
            "Fill or drop NaNs before calling this function."
        )
    valid_tails = {"upper", "lower"}
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {sorted(valid_tails)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if predictor_col not in df.columns:
        raise KeyError(f"predictor_col {predictor_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[predictor_col]):
        raise ValueError(f"predictor_col {predictor_col!r} must be numeric.")
    rng = np.random.default_rng(random_state)
    result = _with_nullable_bools(df, [target_col])
    predictor = pd.to_numeric(df[predictor_col], errors="coerce")
    thresh = threshold if threshold is not None else float(predictor.median())
    if tail == "upper":
        prob = np.where(predictor > thresh, frac * 2, frac * 0.5)
    else:
        prob = np.where(predictor < thresh, frac * 2, frac * 0.5)
    prob = np.clip(prob, 0, 1)
    missing_mask = rng.random(len(df)) < prob
    result.loc[missing_mask, target_col] = np.nan
    return result


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
        Base missingness fraction, strictly between 0 and 1.
    target_col : str
        Numeric column whose own values drive its missingness.
    threshold : float, optional
        Split point for *target_col*. Defaults to its median.
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

    Notes
    -----
    bool columns are returned as nullable ``boolean`` so they can hold missing
    values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    >>> out = simulate_mnar(df, 0.5, target_col="y", random_state=0)
    >>> bool(out["y"].isna().any())
    True
    """
    valid_tails = {"upper", "lower"}
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {sorted(valid_tails)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"target_col {target_col!r} must be numeric for MNAR simulation.")
    rng = np.random.default_rng(random_state)
    result = _with_nullable_bools(df, [target_col])
    col_vals = pd.to_numeric(df[target_col], errors="coerce")
    thresh = threshold if threshold is not None else float(col_vals.median())
    if tail == "upper":
        prob = np.where(col_vals > thresh, frac * 2, frac * 0.25)
    else:
        prob = np.where(col_vals < thresh, frac * 2, frac * 0.25)
    prob = np.clip(prob, 0, 1)
    missing_mask = rng.random(len(df)) < prob
    result.loc[missing_mask, target_col] = np.nan
    return result


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
        If *df* already contains missing values or a mechanism is unknown.

    Notes
    -----
    bool columns are returned as nullable ``boolean`` so they can hold missing
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

    result = df.copy()
    for i, spec in enumerate(mechanisms):
        spec = dict(spec)
        mtype = spec.pop("mechanism", None)
        if mtype is None or mtype.lower() not in _sim_dispatch:
            raise ValueError(
                f"unknown mechanism {mtype!r}. "
                f"Choose from {sorted(_sim_dispatch)}."
            )
        seed = (random_state + i) if random_state is not None else None
        result = _sim_dispatch[mtype.lower()](result, random_state=seed, **spec)
    return result


# ---------------------------------------------------------------------------
# Internal "raw" helpers used by simulate_mixed (no existing-NaN guard)
# ---------------------------------------------------------------------------

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
    valid_tails = {"upper", "lower"}
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {sorted(valid_tails)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    rng = np.random.default_rng(random_state)
    result = _with_nullable_bools(df, [target_col])
    predictor = pd.to_numeric(df[predictor_col], errors="coerce")
    thresh = threshold if threshold is not None else float(predictor.median())
    if tail == "upper":
        prob = np.where(predictor > thresh, frac * 2, frac * 0.5)
    else:
        prob = np.where(predictor < thresh, frac * 2, frac * 0.5)
    prob = np.clip(prob, 0, 1)
    already_missing = result[target_col].isnull()
    prob = np.where(already_missing, 0.0, prob)
    missing_mask = rng.random(len(df)) < prob
    result.loc[missing_mask, target_col] = np.nan
    return result


def _simulate_mnar_raw(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    threshold: Optional[float] = None,
    tail: str = "upper",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    valid_tails = {"upper", "lower"}
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {sorted(valid_tails)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    rng = np.random.default_rng(random_state)
    result = _with_nullable_bools(df, [target_col])
    col_vals = pd.to_numeric(df[target_col], errors="coerce")
    thresh = threshold if threshold is not None else float(col_vals.median())
    if tail == "upper":
        prob = np.where(col_vals > thresh, frac * 2, frac * 0.25)
    else:
        prob = np.where(col_vals < thresh, frac * 2, frac * 0.25)
    prob = np.clip(prob, 0, 1)
    already_missing = result[target_col].isnull()
    prob = np.where(already_missing, 0.0, prob)
    missing_mask = rng.random(len(df)) < prob
    result.loc[missing_mask, target_col] = np.nan
    return result
