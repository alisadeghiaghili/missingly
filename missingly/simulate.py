"""Missing data simulation utilities."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def simulate_mcar(
    df: pd.DataFrame,
    frac: float = 0.1,
    *,
    columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce MCAR (Missing Completely At Random) missingness."""
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
    result = df.copy()
    target_cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
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
    """Introduce MAR (Missing At Random) missingness."""
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
    result = df.copy()
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
    """Introduce MNAR (Missing Not At Random) missingness."""
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
    result = df.copy()
    target_cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
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
    result = df.copy()
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
    result = df.copy()
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
