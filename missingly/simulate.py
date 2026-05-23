"""Missing data simulation utilities (experimental).

.. deprecated:: 0.2.0
    These functions are experimental and may be moved to a dedicated
    benchmarking package (e.g., ``missingly-simulate``) in a future release.
    They emit :class:`FutureWarning` on every call.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._deprecation import deprecated_api


@deprecated_api(
    "Simulation utilities are experimental and may move to a dedicated benchmarking package in a future release.",
    since="0.2.0",
)
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
        Complete (or near-complete) DataFrame to corrupt.
    frac : float, default 0.1
        Fraction of values per target column to set as missing (0 < frac < 1).
    columns : list of str, optional
        Columns to corrupt. Defaults to all numeric columns.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values injected at random positions.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")

    rng = np.random.default_rng(random_state)
    result = df.copy()
    target_cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()

    for col in target_cols:
        n = len(result)
        n_missing = max(1, int(round(n * frac)))
        indices = rng.choice(n, size=n_missing, replace=False)
        result.iloc[indices, result.columns.get_loc(col)] = np.nan

    return result


@deprecated_api(
    "Simulation utilities are experimental and may move to a dedicated benchmarking package in a future release.",
    since="0.2.0",
)
def simulate_mar(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    target_col: str,
    predictor_col: str,
    threshold: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce MAR (Missing At Random) missingness."""
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if predictor_col not in df.columns:
        raise KeyError(f"predictor_col {predictor_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[predictor_col]):
        raise TypeError(f"predictor_col {predictor_col!r} must be numeric.")

    rng = np.random.default_rng(random_state)
    result = df.copy()
    predictor = pd.to_numeric(df[predictor_col], errors="coerce")
    thresh = threshold if threshold is not None else float(predictor.median())

    prob = np.where(predictor > thresh, frac * 2, frac * 0.5)
    prob = np.clip(prob, 0, 1)

    missing_mask = rng.random(len(df)) < prob
    result.loc[missing_mask, target_col] = np.nan
    return result


@deprecated_api(
    "Simulation utilities are experimental and may move to a dedicated benchmarking package in a future release.",
    since="0.2.0",
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
        Input DataFrame.
    frac : float, default 0.1
        Approximate fraction of *target_col* to set missing.
    target_col : str
        Column in which to inject NaN values.
    threshold : float, optional
        Decision boundary. Defaults to the median of *target_col*.
    tail : {"upper", "lower"}, default "upper"
        ``"upper"`` makes values *above* threshold more likely missing;
        ``"lower"`` makes values *below* threshold more likely missing.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with MNAR NaN values in *target_col*.
    """
    valid_tails = {"upper", "lower"}
    if tail not in valid_tails:
        raise ValueError(f"tail must be one of {sorted(valid_tails)}; got {tail!r}.")
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1); got {frac!r}.")
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"target_col {target_col!r} must be numeric for MNAR simulation.")

    rng = np.random.default_rng(random_state)
    result = df.copy()
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


@deprecated_api(
    "Simulation utilities are experimental and may move to a dedicated benchmarking package in a future release.",
    since="0.2.0",
)
def simulate_mixed(
    df: pd.DataFrame,
    mechanisms: List[Dict],
    *,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Apply a mixture of MCAR / MAR / MNAR mechanisms.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    mechanisms : list of dict
        Each dict specifies one corruption step. Required keys:

        * ``"type"`` : ``"mcar"`` | ``"mar"`` | ``"mnar"``
        * ``"frac"`` : float
        * All other keyword arguments accepted by the corresponding
          ``simulate_*`` function (excluding ``random_state``).

    random_state : int, optional
        Base seed; each step increments the seed by 1 for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the requested missingness patterns applied
        sequentially.
    """
    if df.isnull().any().any():
        raise ValueError(
            "simulate_mixed expects a complete DataFrame (no existing missing values). "
            "Fill or drop NaNs before calling this function."
        )

    _sim_map = {
        "mcar": simulate_mcar.__wrapped__,
        "mar": simulate_mar.__wrapped__,
        "mnar": simulate_mnar.__wrapped__,
    }

    result = df.copy()
    for i, spec in enumerate(mechanisms):
        # Use a copy so we don't mutate the caller's dict
        spec = dict(spec)
        mtype = spec.pop("type", None)
        if mtype not in _sim_map:
            raise ValueError(
                f"Unknown mechanism type {mtype!r}. "
                f"Choose from {sorted(_sim_map)}."
            )
        seed = (random_state + i) if random_state is not None else None
        result = _sim_map[mtype](result, random_state=seed, **spec)

    return result
