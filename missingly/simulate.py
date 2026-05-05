"""Missing data simulation utilities.

This module provides tools for artificially introducing missing values into
a complete DataFrame under three well-defined statistical mechanisms:

* **MCAR** (:func:`simulate_mcar`) — Missing Completely At Random.
  Each value is independently masked with probability *p*, regardless of
  any other value in the dataset.  The gold-standard null model.

* **MAR** (:func:`simulate_mar`) — Missing At Random.
  The probability of a value being masked depends on *other observed*
  columns, not on the missing value itself.  Missingness is predictable
  from the rest of the data.

* **MNAR** (:func:`simulate_mnar`) — Missing Not At Random.
  The probability of a value being masked depends on the *value itself*
  (e.g. high earners are less likely to report income).  The hardest
  mechanism to correct for in practice.

* :func:`simulate_mixed` — Convenience wrapper that applies different
  mechanisms to different columns in a single call.

All functions
-------------
* Accept a **complete** DataFrame (no pre-existing NaNs).
* Return a copy — the input is never mutated.
* Are reproducible via ``random_state``.
* Preserve dtypes of unmasked values.

Typical use-cases
-----------------
* Benchmarking imputation methods under controlled conditions.
* Evaluating :func:`~missingly.stats.diagnose_missing` on synthetic data
  with a known ground truth mechanism.
* Teaching / demonstration.

Compatibility
-------------
Requires Python 3.9+ and pandas 2.0+.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_complete(df: pd.DataFrame, fn_name: str) -> None:
    """Raise ValueError if *df* contains any NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    fn_name : str
        Name of the calling function (used in the error message).

    Raises
    ------
    ValueError
        If *df* has missing values.
    """
    if df.isnull().any().any():
        raise ValueError(
            f"{fn_name}() requires a complete DataFrame (no missing values). "
            "Drop or impute existing NaNs before simulating."
        )


def _validate_frac(frac: float, name: str = "frac") -> None:
    """Raise ValueError if *frac* is not in (0, 1].

    Parameters
    ----------
    frac : float
        Fraction to validate.
    name : str
        Parameter name used in the error message.

    Raises
    ------
    ValueError
        If *frac* is outside (0, 1].
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"{name} must be in (0, 1]; got {frac!r}")


def _resolve_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]],
) -> List[str]:
    """Return the list of columns to operate on.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str or None
        If None, all columns are returned.

    Returns
    -------
    list of str

    Raises
    ------
    ValueError
        If any column in *columns* is not present in *df*.
    """
    if columns is None:
        return df.columns.tolist()
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found in DataFrame: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return list(columns)


# ---------------------------------------------------------------------------
# MCAR
# ---------------------------------------------------------------------------

def simulate_mcar(
    df: pd.DataFrame,
    frac: float = 0.10,
    columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce Missing Completely At Random (MCAR) values.

    Each cell in the target columns is independently masked with
    probability *frac*, regardless of any other value in the dataset.
    This is the simplest and most benign missing-data mechanism.

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** DataFrame (no missing values).
    frac : float, optional
        Fraction of values to mask per column (default 0.10).
        Must be in (0, 1].
    columns : list of str, optional
        Columns to introduce missingness into.  Defaults to all columns.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values introduced under MCAR.

    Raises
    ------
    ValueError
        If *df* has pre-existing missing values.
    ValueError
        If *frac* is not in (0, 1].

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
    >>> df_miss = simulate_mcar(df, frac=0.20, random_state=42)
    >>> df_miss.isnull().mean().round(2)
    a    0.2
    b    0.2
    dtype: float64
    """
    _validate_complete(df, "simulate_mcar")
    _validate_frac(frac, "frac")
    cols = _resolve_columns(df, columns)

    rng = np.random.default_rng(random_state)
    result = df.copy()
    n = len(df)

    for col in cols:
        n_mask = max(1, int(round(n * frac)))
        idx = rng.choice(df.index, size=n_mask, replace=False)
        result.loc[idx, col] = np.nan

    return result


# ---------------------------------------------------------------------------
# MAR
# ---------------------------------------------------------------------------

def simulate_mar(
    df: pd.DataFrame,
    target_col: str,
    predictor_col: str,
    frac: float = 0.10,
    tail: str = "upper",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce Missing At Random (MAR) values in one column.

    The probability of missingness in *target_col* depends on the value
    of *predictor_col*: rows where *predictor_col* is in the upper (or
    lower) tail are preferentially masked.  This simulates the realistic
    scenario where missingness is explainable by other observed variables.

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** DataFrame (no missing values).
    target_col : str
        Column in which NaN values will be introduced.
    predictor_col : str
        Numeric column whose values determine missingness probability.
        High (or low) values of *predictor_col* increase the chance that
        the corresponding row in *target_col* is masked.
    frac : float, optional
        Overall fraction of rows in *target_col* to mask (default 0.10).
        Must be in (0, 1].
    tail : {'upper', 'lower'}, optional
        Whether rows with *high* (``'upper'``) or *low* (``'lower'``)
        values of *predictor_col* are preferentially masked.
        Default ``'upper'``.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values introduced under MAR.

    Raises
    ------
    ValueError
        If *df* has pre-existing missing values.
    ValueError
        If *frac* is not in (0, 1].
    ValueError
        If *target_col* or *predictor_col* is not in *df*.
    ValueError
        If *tail* is not ``'upper'`` or ``'lower'``.
    ValueError
        If *predictor_col* is not numeric.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     'income': rng.normal(50000, 10000, 200),
    ...     'age':    rng.integers(20, 65, 200),
    ... })
    >>> # Mask 'age' preferentially for high-income rows (MAR)
    >>> df_miss = simulate_mar(df, target_col='age',
    ...                        predictor_col='income', frac=0.15)
    """
    _validate_complete(df, "simulate_mar")
    _validate_frac(frac, "frac")
    _resolve_columns(df, [target_col, predictor_col])

    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower'; got {tail!r}")
    if not pd.api.types.is_numeric_dtype(df[predictor_col]):
        raise ValueError(
            f"predictor_col '{predictor_col}' must be numeric; "
            f"got dtype '{df[predictor_col].dtype}'."
        )

    rng = np.random.default_rng(random_state)
    result = df.copy()
    n = len(df)
    n_mask = max(1, int(round(n * frac)))

    # Build weights: rows with high (or low) predictor values get higher weight
    vals = df[predictor_col].to_numpy(dtype=float)
    # Rank-based weights so the result is robust to outliers
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    if tail == "upper":
        weights = ranks
    else:
        weights = ranks.max() + 1 - ranks

    weights = weights / weights.sum()
    chosen = rng.choice(df.index, size=n_mask, replace=False, p=weights)
    result.loc[chosen, target_col] = np.nan

    return result


# ---------------------------------------------------------------------------
# MNAR
# ---------------------------------------------------------------------------

def simulate_mnar(
    df: pd.DataFrame,
    target_col: str,
    frac: float = 0.10,
    tail: str = "upper",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Introduce Missing Not At Random (MNAR) values in one column.

    The probability of missingness depends on the *value of the column
    itself*: rows with high (or low) values of *target_col* are
    preferentially masked.  This is the hardest mechanism to handle
    statistically because the bias is unobservable from the data alone.

    Common real-world examples:

    * High earners are less likely to report income → MNAR upper tail.
    * Patients who feel worst are less likely to respond to follow-up
      surveys → MNAR lower tail.

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** DataFrame (no missing values).
    target_col : str
        Column in which NaN values will be introduced.  Must be numeric.
    frac : float, optional
        Overall fraction of rows to mask (default 0.10).
        Must be in (0, 1].
    tail : {'upper', 'lower'}, optional
        Whether rows with *high* (``'upper'``) or *low* (``'lower'``)
        values of *target_col* are preferentially masked.
        Default ``'upper'``.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values introduced under MNAR.

    Raises
    ------
    ValueError
        If *df* has pre-existing missing values.
    ValueError
        If *frac* is not in (0, 1].
    ValueError
        If *target_col* is not in *df* or is not numeric.
    ValueError
        If *tail* is not ``'upper'`` or ``'lower'``.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'income': rng.normal(50000, 10000, 200)})
    >>> # High earners are less likely to report income
    >>> df_miss = simulate_mnar(df, target_col='income',
    ...                         frac=0.15, tail='upper')
    """
    _validate_complete(df, "simulate_mnar")
    _validate_frac(frac, "frac")
    _resolve_columns(df, [target_col])

    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower'; got {tail!r}")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"target_col '{target_col}' must be numeric for MNAR simulation; "
            f"got dtype '{df[target_col].dtype}'."
        )

    rng = np.random.default_rng(random_state)
    result = df.copy()
    n = len(df)
    n_mask = max(1, int(round(n * frac)))

    vals = df[target_col].to_numpy(dtype=float)
    ranks = pd.Series(vals).rank(method="average").to_numpy()
    if tail == "upper":
        weights = ranks
    else:
        weights = ranks.max() + 1 - ranks

    weights = weights / weights.sum()
    chosen = rng.choice(df.index, size=n_mask, replace=False, p=weights)
    result.loc[chosen, target_col] = np.nan

    return result


# ---------------------------------------------------------------------------
# Mixed convenience wrapper
# ---------------------------------------------------------------------------

def simulate_mixed(
    df: pd.DataFrame,
    spec: List[Dict],
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Apply different missing-data mechanisms to different columns.

    Sequentially applies MCAR, MAR, or MNAR simulation according to a
    list of specification dicts.  Each spec dict describes one
    missingness injection step.  Steps are applied in order.

    Parameters
    ----------
    df : pd.DataFrame
        A **complete** DataFrame (no missing values).
    spec : list of dict
        Each dict must have a ``'mechanism'`` key
        (``'MCAR'``, ``'MAR'``, or ``'MNAR'``) plus the keyword
        arguments for the corresponding simulate function.  Do *not*
        include ``random_state`` in the spec dicts — use the top-level
        ``random_state`` parameter instead.

        MCAR dict keys: ``columns`` (optional), ``frac``.
        MAR  dict keys: ``target_col``, ``predictor_col``, ``frac``,
        ``tail`` (optional).
        MNAR dict keys: ``target_col``, ``frac``, ``tail`` (optional).

    random_state : int or None, optional
        Base random seed.  Each step uses an independent derived seed so
        results are fully reproducible.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values introduced according to *spec*.

    Raises
    ------
    ValueError
        If *df* has pre-existing missing values.
    ValueError
        If any spec dict has an unknown ``'mechanism'`` value.
    ValueError
        Propagated from the underlying simulate functions.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     'age':    rng.integers(20, 65, 300).astype(float),
    ...     'income': rng.normal(50000, 10000, 300),
    ...     'score':  rng.normal(0, 1, 300),
    ... })
    >>> spec = [
    ...     {'mechanism': 'MCAR', 'columns': ['score'], 'frac': 0.10},
    ...     {'mechanism': 'MAR',  'target_col': 'age',
    ...      'predictor_col': 'income', 'frac': 0.15},
    ...     {'mechanism': 'MNAR', 'target_col': 'income',
    ...      'frac': 0.10, 'tail': 'upper'},
    ... ]
    >>> df_miss = simulate_mixed(df, spec, random_state=42)
    """
    _validate_complete(df, "simulate_mixed")

    _VALID_MECHANISMS = frozenset({"MCAR", "MAR", "MNAR"})
    for i, s in enumerate(spec):
        mech = s.get("mechanism", "")
        if mech not in _VALID_MECHANISMS:
            raise ValueError(
                f"spec[{i}]: unknown mechanism {mech!r}. "
                f"Valid: {sorted(_VALID_MECHANISMS)}"
            )

    rng = np.random.default_rng(random_state)
    result = df.copy()

    for i, s in enumerate(spec):
        step_seed = int(rng.integers(0, 2**31))
        s_kwargs = {k: v for k, v in s.items() if k != "mechanism"}
        mech = s["mechanism"]

        if mech == "MCAR":
            result = simulate_mcar(result, random_state=step_seed, **s_kwargs)
        elif mech == "MAR":
            result = simulate_mar(result, random_state=step_seed, **s_kwargs)
        else:  # MNAR
            result = simulate_mnar(result, random_state=step_seed, **s_kwargs)

    return result
