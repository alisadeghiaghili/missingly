"""Time series missing data analysis for missingly.

This module provides diagnostic and imputation utilities designed
specifically for time-indexed DataFrames.

All public functions assume the DataFrame's index is either a
``DatetimIndex`` or an ordinal integer index representing ordered
observations (rows must be in chronological order).

Functions
---------
miss_ts_summary
    Per-column summary of gaps: count, mean length, max length, etc.
gap_table
    Tidy table of every individual gap per column (start, end, length).
vis_ts_miss
    Tile heatmap of missingness along the time axis.
vis_gap_lengths
    Distribution (histogram or box) of gap lengths per column.
vis_miss_over_time
    Rolling-window missingness rate line chart for all columns.
impute_ts
    Time-series-aware imputation: forward-fill, backward-fill,
    linear interpolation, time interpolation, and spline.
"""

from __future__ import annotations

import inspect
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from ._deprecation import deprecated_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sorted(df: pd.DataFrame) -> None:
    """Raise ValueError if the DataFrame index is not monotonically increasing."""
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be monotonically increasing (sorted). "
            "Call df.sort_index() before using time-series functions."
        )


def _nullity(df: pd.DataFrame, missing_values: Optional[List] = None) -> pd.DataFrame:
    """Return a boolean DataFrame: True where a value is missing."""
    null_mask = df.isnull()
    if missing_values:
        null_mask = null_mask | df.isin(missing_values)
    return null_mask


def _gaps_for_column(
    series: pd.Series,
    missing_values: Optional[List] = None,
) -> List[dict]:
    """Return a list of gap dicts for a single Series.

    Each dict has keys: ``start``, ``end``, ``length``.
    """
    null_mask = series.isnull()
    if missing_values:
        null_mask = null_mask | series.isin(missing_values)

    gaps = []
    in_gap = False
    gap_start = None

    for idx, is_null in zip(series.index, null_mask):
        if is_null and not in_gap:
            in_gap = True
            gap_start = idx
        elif not is_null and in_gap:
            in_gap = False
            gaps.append({"start": gap_start, "end": idx, "length": None})

    if in_gap:
        gaps.append({"start": gap_start, "end": series.index[-1], "length": None})

    # Compute lengths as index differences where possible
    for gap in gaps:
        try:
            length = series.index.get_loc(gap["end"]) - series.index.get_loc(gap["start"])
        except Exception:
            length = None
        gap["length"] = length

    return gaps


def _boxplot_compat(ax, data: list, labels: list, **kwargs) -> None:
    """Draw a boxplot, compatible with both old and new matplotlib APIs."""
    sig = inspect.signature(ax.boxplot)
    if "tick_labels" in sig.parameters:
        ax.boxplot(data, tick_labels=labels, **kwargs)
    else:
        ax.boxplot(data, labels=labels, **kwargs)


def miss_ts_summary(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Per-column summary of missing gaps in a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    missing_values : list, optional
        Additional sentinel values to treat as missing (e.g. ``[-1, "N/A"]``).

    Returns
    -------
    pd.DataFrame
        One row per column with columns:

        * ``n_miss``      — total missing observations
        * ``pct_miss``    — percentage missing
        * ``n_gaps``      — number of contiguous missing runs
        * ``mean_gap``    — mean gap length (in index units)
        * ``max_gap``     — maximum gap length
        * ``median_gap``  — median gap length

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=10, freq="D")
    >>> df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, 5, np.nan, 7, 8, 9, 10]}, index=idx)
    >>> miss_ts_summary(df)
    """
    _require_sorted(df)
    records = []
    for col in df.columns:
        null_mask = df[col].isnull()
        if missing_values:
            null_mask = null_mask | df[col].isin(missing_values)

        n_miss = int(null_mask.sum())
        pct_miss = round(100.0 * n_miss / len(df), 2) if len(df) > 0 else 0.0
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        lengths = [g["length"] for g in gaps if g["length"] is not None]

        records.append(
            {
                "variable": col,
                "n_miss": n_miss,
                "pct_miss": pct_miss,
                "n_gaps": len(gaps),
                "mean_gap": round(float(np.mean(lengths)), 2) if lengths else 0.0,
                "max_gap": int(max(lengths)) if lengths else 0,
                "median_gap": float(np.median(lengths)) if lengths else 0.0,
            }
        )
    return pd.DataFrame(records).set_index("variable")


@deprecated_api(
    "gap_table is an experimental utility and may be moved to a separate timeseries package in a future release.",
    since="0.2.0",
)
def gap_table(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Tidy table of every individual missing gap per column (experimental).

    .. deprecated:: 0.2.0
        Experimental — may be moved or renamed in a future release.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    missing_values : list, optional
        Additional sentinel values to treat as missing.
    columns : list of str, optional
        Subset of columns to analyse. Defaults to all columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``variable``, ``gap_id``, ``start``, ``end``, ``length``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=8, freq="D")
    >>> df = pd.DataFrame({"x": [1, np.nan, np.nan, 4, np.nan, 6, 7, 8]}, index=idx)
    >>> gap_table(df)
    """
    _require_sorted(df)
    target_cols = columns if columns is not None else df.columns.tolist()
    rows = []
    for col in target_cols:
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        for i, gap in enumerate(gaps, start=1):
            rows.append(
                {
                    "variable": col,
                    "gap_id": i,
                    "start": gap["start"],
                    "end": gap["end"],
                    "length": gap["length"],
                }
            )
    return pd.DataFrame(rows)


def vis_ts_miss(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "RdYlGn",
    title: str = "Time Series Missingness",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Tile heatmap showing missingness along the time axis.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    missing_values : list, optional
        Additional sentinel values to treat as missing.
    figsize : tuple, optional
        Figure size. Defaults to ``(12, max(3, n_cols * 0.5))``.
    cmap : str, default ``"RdYlGn"``
        Colormap; green = present, red = missing.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_sorted(df)
    null_mask = _nullity(df, missing_values=missing_values)
    present = (~null_mask).astype(int)

    n_cols = len(df.columns)
    if figsize is None:
        figsize = (12, max(3, n_cols * 0.5))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        present.T.values,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(range(n_cols))
    ax.set_yticklabels(df.columns)

    n_ticks = min(10, len(df))
    tick_indices = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [str(df.index[i])[:10] for i in tick_indices],
        rotation=45,
        ha="right",
    )

    present_patch = mpatches.Patch(color=plt.get_cmap(cmap)(1.0), label="Present")
    missing_patch = mpatches.Patch(color=plt.get_cmap(cmap)(0.0), label="Missing")
    ax.legend(handles=[present_patch, missing_patch], loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    return ax


@deprecated_api(
    "vis_gap_lengths is an experimental utility and may be moved to a separate timeseries package in a future release.",
    since="0.2.0",
)
def vis_gap_lengths(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    kind: Literal["hist", "box"] = "hist",
    columns: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    title: str = "Gap Length Distribution",
) -> matplotlib.figure.Figure:
    """Distribution of gap lengths per column (experimental).

    .. deprecated:: 0.2.0
        Experimental — may be moved or renamed in a future release.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    missing_values : list, optional
        Additional sentinel values to treat as missing.
    kind : {"hist", "box"}, default "hist"
        Plot type.
    columns : list of str, optional
        Subset of columns to analyse.
    figsize : tuple, optional
        Figure size.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_sorted(df)
    target_cols = columns if columns is not None else df.columns.tolist()

    col_gaps: dict = {}
    for col in target_cols:
        gaps = _gaps_for_column(df[col], missing_values=missing_values)
        lengths = [g["length"] for g in gaps if g["length"] is not None and g["length"] > 0]
        if lengths:
            col_gaps[col] = lengths

    if not col_gaps:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No gaps found", ha="center", va="center")
        ax.set_title(title)
        return fig

    n = len(col_gaps)
    if figsize is None:
        figsize = (max(6, n * 2), 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for ax, (col, lengths) in zip(axes[0], col_gaps.items()):
        if kind == "hist":
            ax.hist(lengths, bins="auto", color="steelblue", edgecolor="white")
            ax.set_xlabel("Gap length")
            ax.set_ylabel("Count")
        else:
            _boxplot_compat(ax, [lengths], [col])
            ax.set_ylabel("Gap length")
        ax.set_title(col)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def vis_miss_over_time(
    df: pd.DataFrame,
    *,
    missing_values: Optional[List] = None,
    window: int = 10,
    figsize: Optional[tuple] = None,
    title: str = "Rolling Missingness Rate",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Rolling-window missingness rate line chart.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    missing_values : list, optional
        Additional sentinel values to treat as missing.
    window : int, default 10
        Rolling window size (in rows).
    figsize : tuple, optional
        Figure size. Defaults to ``(12, 4)``.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_sorted(df)
    null_mask = _nullity(df, missing_values=missing_values).astype(float)
    rolling_rate = null_mask.rolling(window=window, min_periods=1).mean() * 100

    if figsize is None:
        figsize = (12, 4)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for col in rolling_rate.columns:
        ax.plot(rolling_rate.index, rolling_rate[col], label=col)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"% missing (window={window})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return ax


def impute_ts(
    df: pd.DataFrame,
    method: Literal["ffill", "bfill", "linear", "time", "spline"] = "linear",
    *,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    spline_order: int = 3,
    missing_values: Optional[List] = None,
) -> pd.DataFrame:
    """Time-series-aware imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame (index must be sorted).
    method : {"ffill", "bfill", "linear", "time", "spline"}, default "linear"
        Imputation strategy:

        * ``"ffill"``   — forward-fill (last observation carried forward)
        * ``"bfill"``   — backward-fill (next observation carried backward)
        * ``"linear"``  — linear interpolation by row position
        * ``"time"``    — linear interpolation weighted by time index
        * ``"spline"``  — spline interpolation (requires ``spline_order``)

    columns : list of str, optional
        Subset of columns to impute. Defaults to all numeric columns with
        missing values.
    limit : int, optional
        Maximum number of consecutive NaNs to fill.
    spline_order : int, default 3
        Order of the spline (only used when ``method="spline"``).
    missing_values : list, optional
        Additional sentinel values to replace with NaN before imputing.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with imputed values.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=5, freq="D")
    >>> df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0, 5.0]}, index=idx)
    >>> impute_ts(df, method="linear")
    """
    _require_sorted(df)
    valid_methods = {"ffill", "bfill", "linear", "time", "spline"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}; got {method!r}."
        )

    result = df.copy()

    if missing_values:
        result = result.replace(missing_values, np.nan)

    if columns is not None:
        missing_cols = [c for c in columns if c not in result.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        target_cols = columns
    else:
        target_cols = [
            c for c in result.columns
            if result[c].isnull().any() and pd.api.types.is_numeric_dtype(result[c])
        ]

    for col in target_cols:
        if method in ("ffill", "bfill"):
            result[col] = result[col].ffill(limit=limit) if method == "ffill" else result[col].bfill(limit=limit)
        elif method == "spline":
            result[col] = result[col].interpolate(
                method="spline", order=spline_order, limit=limit
            )
        else:
            result[col] = result[col].interpolate(method=method, limit=limit)

    return result
